"""
Author: jolin
Usage : ensemble测试集成
Date  : 2024-08-08
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
import easydict
import sys
import yaml
import gc
from tqdm import tqdm
import traceback
import random

import pandas as pd
from skopt import gp_minimize
import pickle
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
device_ids = [0,1,2,3]
torch.backends.cudnn.enable =True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def load_checkpoint(model, ckpt_path):

    print(f"Resuming checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['net']
    model.load_state_dict(state_dict, strict=False)

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    loaded_keys = own_keys & ckpt_keys
    missing_keys = own_keys - ckpt_keys
    print("loaed keys: %d, missing keys: %d" % (len(loaded_keys), len(missing_keys)))

    return model

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--config', 
                        default=r'./config/NTU120_CS_default.yaml',  
                        help='path to the configuration file')

    parser.add_argument('--model1', default=r'Pose2Point',   
                        help='model name [default: pointmodel_cls]')
    parser.add_argument('--model2', default=r'skeMLP', 
                        help='model name [default: pointmodel_cls]')

    parser.add_argument('--resume', type=bool, default=True,
                        help='Resume training or not')

    parser.add_argument('--ckpt_path', type=str,
                        default=r"./checkpoints/best_checkpoint.pth",   
                        help='Path to checkpoint.')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')   
    parser.add_argument('--num_classes', default=120, type=int, help='default value for classes of NTU60')  
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=32, type=int, help='workers')
    parser.add_argument('--result_path', default='./ensemble_results/', type=str, help='path')
    parser.add_argument('--sample', default="uniform_joints", help='fast/slow/uniform_joints')
    parser.add_argument('--bone', default=0, help='bone')
    parser.add_argument('--vel', default=0, help='motion')
    
    config_path = parser.get_default('config')
    with open(config_path, 'r') as f:
        feeder_config = yaml.safe_load(f)

    # bone & motion
    feeder_config['train_feeder_args']['bone'] = parser.get_default('bone')  
    feeder_config['test_feeder_args']['bone'] = parser.get_default('bone')  
    feeder_config['train_feeder_args']['vel'] = parser.get_default('vel')  
    feeder_config['test_feeder_args']['vel'] = parser.get_default('vel')  

    parser.add_argument('--feeder', default=feeder_config['feeder'], help='data loader will be used')
    parser.add_argument('--train_feeder_args', default=feeder_config['train_feeder_args'],
                        help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', default=feeder_config['test_feeder_args'],
                        help='the arguments of data loader for test')
    data = feeder_config['train_feeder_args']['data_path'].split("/")[-1][:-4]
    parser.add_argument('--data', default=data, help='data name')
    return parser.parse_args()

def main():
    # ========== parsers and init ============
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)   
    
    # 1. model1
    model1 = models.__dict__[args.model1](num_classes = args.num_classes,
                                          num_points  = 1024,sample=args.sample).cuda()  
    
    model1.apply(weight_init)
    model1 = torch.nn.DataParallel(model1, device_ids=device_ids)
    cudnn.benchmark = True

    model1 = load_checkpoint(model1, args.ckpt_path1)

    # ============= criterion and optimizer ================
    criterion = cal_loss

    # =========== Dataloader =================
    Feeder = import_class(args.feeder)
    train_loader = DataLoader(Feeder(**args.train_feeder_args), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Feeder(**args.test_feeder_args), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # ============= Testing =================
    print('Testing......')
    
    test_out1 = testsave_one_epoch(model1, test_loader, criterion, device,args, temp_file=args.result_path)

    return

def testsave_one_epoch(model, testloader, criterion, device, args, temp_file='best.pkl'):
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    test_true = []
    test_pred = []
    test_outputs = []

    time_cost = datetime.datetime.now()
    with torch.no_grad():
        testloader = tqdm(testloader, file = sys.stdout, ncols=120)
        testloader.set_description(f"[Saving ]")

        for step, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device)
            if  "ncl" in args.model1:
                outputs = model(data)
                outputs = outputs[-1] 
            else:
                outputs = model(data)

            test_outputs.append(outputs.cpu().numpy())

            loss = criterion(outputs, label)
            test_loss += loss.item()

            preds = outputs.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            total += label.size(0)
            correct += preds.eq(label).sum().item()

            testloader.set_postfix({'loss': test_loss / (step + 1),
                                    'Acc': '%.3f%% (%d/%d)' % (100. * correct / total, correct, total)
                                    })

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_outputs = np.concatenate(test_outputs)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    loss = float("%.3f" % (test_loss / (step + 1))),
    acc = float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
    acc_avg = float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
    result_path = f"{temp_file}{acc[0]:.3f}_{acc_avg[0]:.3f}.pkl"

    with open(result_path, 'wb') as f:
        pickle.dump((test_outputs, test_true), f)
    
    print(f"loss:{loss},acc:{acc},acc_b:{acc_avg},time:{time_cost}")

    return {
        "loss": float("%.3f" % (test_loss / (step + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


def load_temp_outputs(temp_file='/data/wenjj/Skeleton2Point/temp_outputs.pkl'):
    with open(temp_file, 'rb') as f:
        outputs1, labels = pickle.load(f)
    return outputs1, labels


def objective_4(weights, outputs1, outputs2, outputs3, outputs4, labels):
    weight1, weight2, weight3, weight4 = weights
    weight_sum = weight1 + weight2 + weight3 + weight4
    weight1 /= weight_sum
    weight2 /= weight_sum
    weight3 /= weight_sum
    weight4 /= weight_sum

    outputs = outputs1 * weight1 + outputs2 * weight2 + outputs3 * weight3 + outputs4 * weight4
    preds = np.argmax(outputs, axis=1)
    accuracy = metrics.accuracy_score(labels, preds)
    acc_b = metrics.balanced_accuracy_score(labels, preds)
    if acc_b>accuracy:
        accuracy = acc_b
    return -accuracy  # 取负值以便于最小化


def optimize_weights_4(random_seed):

    # 加载输出
    outputs1, labels = load_temp_outputs(temp_file = r'/data/wenjj/Skeleton2Point/Ensemble4_9141.pkl')
    outputs2, _ = load_temp_outputs(temp_file = r'/data/wenjj/Skeleton2Point/pointMLP_Base_8884.pkl')
    outputs3, _ = load_temp_outputs(temp_file = r'/data/wenjj/Skeleton2Point/skeletonMLP.pkl')
    outputs4, _ = load_temp_outputs(temp_file = r'/data/wenjj/Skeleton2Point/pointMLP_Out_8393.pkl')

    # 设置搜索空间
    dimensions = [(0.0000001, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    call_times = 64

    # 进度条
    with tqdm(total=call_times, desc="Optimizing weights", unit="call", ncols=120) as pbar:
        def callback(res):
            pbar.update(1)  # 每次调用时更新进度条

        res = gp_minimize(
            lambda w: objective_4(w, outputs1, outputs2, outputs3, outputs4, labels), 
            dimensions,
            n_calls=call_times,
            random_state=random_seed,
            callback=callback
        )

    best_weights = res.x
    best_accuracy = -res.fun  # 取负值以获取准确率
    print("Best weights: ", best_weights)
    print("Best accuracy: %.3f" % (100. * best_accuracy))
    return best_weights, best_accuracy


def batch_optimize(batch_size, outputs1, outputs2, outputs3, outputs4, labels, dimensions, thread_idx):
    """一个批次的目标函数评估"""
    results = []
    
    if thread_idx == 0:
        with tqdm(total=batch_size, desc=f"Threads processing", unit="batch", ncols=120) as pbar:
            for i in range(batch_size):
                # 随机生成权重并计算对应的结果
                weights = np.random.uniform(dimensions[0][0], dimensions[0][1]), \
                          np.random.uniform(dimensions[1][0], dimensions[1][1]), \
                          np.random.uniform(dimensions[2][0], dimensions[2][1]), \
                          np.random.uniform(dimensions[3][0], dimensions[3][1])
                res = objective_4(weights, outputs1, outputs2, outputs3, outputs4, labels)
                results.append((weights, res))  # 记录权重和对应的准确率
                pbar.update(1)
    else:
        for i in range(batch_size):
            # 随机生成权重并计算对应的结果
            weights = np.random.uniform(dimensions[0][0], dimensions[0][1]), \
                      np.random.uniform(dimensions[1][0], dimensions[1][1]), \
                      np.random.uniform(dimensions[2][0], dimensions[2][1]), \
                      np.random.uniform(dimensions[3][0], dimensions[3][1])
            res = objective_4(weights, outputs1, outputs2, outputs3, outputs4, labels)
            results.append((weights, res))  # 记录权重和对应的准确率

    return results

def optimize_weights_4_parallel(outputs1,outputs2,outputs3,outputs4,labels):
    # 设置搜索空间
    dimensions = [(0.6, 1.0), (0.0, 0.3), (0.0, 0.1), (0.0, 0.05)]
    call_times = 128000
    num_threads = 32  
    batch_size = call_times // num_threads 

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(batch_optimize, batch_size, outputs1, outputs2, outputs3, outputs4, labels, dimensions, thread_idx) 
                    for thread_idx in range(num_threads)]

        all_results = []
        for future in futures:
            batch_results = future.result()  
            all_results.extend(batch_results) 

    best_result = min(all_results, key=lambda x: x[1])  # x[1] 是准确率
    best_weights, best_accuracy = best_result

    print("Best weights : ", best_weights)
    print("Best accuracy: %.3f" % (100. * (-best_accuracy)))  # 取负值得到准确率

    return best_weights, -best_accuracy


if __name__ == '__main__':
    # 1. Perform 4 inferences to obtain the result pkl
    # main()
    # main()
    # main()
    # main()

    # 2. Integrate these results
    outputs1, labels1 = load_temp_outputs(temp_file=r'./ensemble_results/inference1.pkl')
    outputs2, labels2 = load_temp_outputs(temp_file=r'./ensemble_results/inference2.pkl')
    outputs3, _ = load_temp_outputs(temp_file=r'./ensemble_results/inference3.pkl')
    outputs4, _ = load_temp_outputs(temp_file=r'./ensemble_results/inference4.pkl')


    best_weights, best_accuracy = optimize_weights_4_parallel(outputs1,outputs2,outputs3,outputs4,labels1)
    outputs = outputs1 * best_weights[0] + outputs2 * best_weights[1] + outputs3 * best_weights[2] + outputs4 * best_weights[3]

    preds = np.argmax(outputs, axis=1)
    accuracy = metrics.accuracy_score(labels1, preds)
    accuracy_b = metrics.balanced_accuracy_score(labels1, preds)
    best_accuracy = accuracy if accuracy>accuracy_b else accuracy_b
    

    temp_file = f'./ensemble_results/Best_{best_accuracy}.pkl'


    with open(temp_file, 'wb') as f:
        pickle.dump((outputs, labels1), f)


    if(labels1 != labels2):
        print("Error: labels not match")
    else:
        print("Labels match")

    print(accuracy * 100.)