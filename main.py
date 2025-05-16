import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="4,5,6,7"
device_ids = [0,1,2,3]
import json
import logging
import datetime
import torch 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
import sys
import yaml
import gc
from tqdm import tqdm
import csv
import pickle
import torch.nn.functional as F
import random
torch.cuda.device_count()

import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
torch.backends.cudnn.enable =True


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def parse_args():
    """Parameters"""

    parser = argparse.ArgumentParser('training')
    parser.add_argument('--config', default=r'./config/NTU120_CS_default.yaml'
                        help='path to the configuration file')
    parser.add_argument('--model', default=r'Pose2Point',   # Pose2Point, Pose2Point_5
                        help='model name [default: pointnet_cls]')
    parser.add_argument('--result_root', default=r'./logs/NTU120CS',
                        help='dir for saving logs')
    parser.add_argument('--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', default=None,    
                        type=str, help='message after checkpoint')
    
    parser.add_argument('--batch_size', type=int, default=48, help='batch size in training')   
    parser.add_argument('--num_classes', default=120, type=int, help='default value for classes of dataset') 
    parser.add_argument('--epoch', default=256, type=int, help='number of epoch in training')  
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')  
    parser.add_argument('--learning_rate', default=0.025, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--resume_checkpoint', default="", help='resume ckpt')
    parser.add_argument('--sample', default="uniform_joints", help='sampling method')
    parser.add_argument('--mixup', default=0.5, help='Probability of mixup')   
    parser.add_argument('--ncls_loss', default=1, help='Multi-CLS Loss')
    parser.add_argument('--random_rot', default=1, help='Rotating augmentation')  
    parser.add_argument('--random_scale_shift', default=0, help='Translation & Scaling augmentation') 
    parser.add_argument('--bone', default=0, help='bone')
    parser.add_argument('--vel', default=0, help='motion')

    # load the default configuration of feeder
    config_path = parser.get_default('config')
    with open(config_path, 'r') as f:
        feeder_config = yaml.safe_load(f)

    feeder_config['train_feeder_args']['random_rot'] = parser.get_default('random_rot')  
    feeder_config['train_feeder_args']['random_scale_shift'] = parser.get_default('random_scale_shift')  
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

def save_configs(args, config_path):
    """save configuration"""
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    return config_path

# add to smooth label
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.cuda().view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss

def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = 'cuda'#Yes
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)#No
    else:
        device = 'cpu'
    time_str = str(datetime.datetime.now().strftime('-%m%d%H%M'))
    message = time_str

    args.checkpoint = './checkpoints/NTU120CS/' + args.model + message + '-' + str(args.seed)  
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)       

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes,num_points=args.num_points,sample=args.sample)
    
    if args.SmoothCrossEntropy:
        criterion = SmoothCrossEntropy()
    else:
        criterion = cal_loss    #cross entropy loss, apply label smoothing if needed

    net = net.to(device)
    if device == 'cuda': #yes
        net = torch.nn.DataParallel(net,device_ids=device_ids)
        cudnn.benchmark = True

    best_test_acc = 0.  # best test accuracy 
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if args.resume_checkpoint:
        resume_checkpoint=args.resume_checkpoint
        printf(f"Resuming last checkpoint from {resume_checkpoint}")
        checkpoint_path = resume_checkpoint

        checkpoint = torch.load(checkpoint_path)
        new_state_dict = checkpoint['net']  
        net.load_state_dict(new_state_dict, strict=False)
        ckpt_keys = set(new_state_dict.keys())
        own_keys = set(net.state_dict().keys())
        loaded_keys = own_keys & ckpt_keys
        missing_keys = own_keys - ckpt_keys 

        net.load_state_dict(new_state_dict, strict=False)
        print("loaed keys: %d, missing keys: %d" % (len(loaded_keys), len(missing_keys)))

        start_epoch = 0 #checkpoint['epoch']
        best_test_acc = 0 #checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = 0 #checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        optimizer_dict = checkpoint['optimizer']


    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])

    printf('==> Preparing data..')

    Feeder = import_class(args.feeder)
    train_loader = DataLoader(Feeder(**args.train_feeder_args), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Feeder(**args.test_feeder_args), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None: #NO
        printf("-----resume optimizer--------")
        optimizer.load_state_dict(optimizer_dict)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.learning_rate
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch= start_epoch - 1) 

    result_path =  os.path.join(args.result_root, args.model + '-' + args.data +  message + '.csv')
    params_path = os.path.join(args.result_root, args.model + '-' + args.data +  message + '.json')

    save_configs(args, params_path)
    header = ['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc']

    with open(result_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for epoch in range(start_epoch, args.epoch):
            printf('Epoch(%d/%s) Learning Rate %s:' % (epoch, args.epoch, optimizer.param_groups[0]['lr']))
            train_out = train_one_epoch(net, train_loader, optimizer, criterion, device, epoch, args.mixup, args.ncls_loss)  # {"loss", "acc", "acc_avg", "time"}
            test_out = validate(net, test_loader, criterion, device, epoch, args.ncls_loss)
            scheduler.step()

            best_acc = test_out["acc"] if test_out["acc"]>test_out["acc_avg"] else test_out["acc_avg"]
            if  best_acc> best_test_acc:    # test_out["acc"]-->best_acc
                # best_test_acc = best_acc # test_out["acc"]
                is_best = True
            else:
                is_best = False
            
            best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
            best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
            best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
            best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
            best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
            best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
            
            save_model(
                net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
                best_test_acc=best_test_acc,  # best test accuracy
                best_train_acc=best_train_acc,
                best_test_acc_avg=best_test_acc_avg,
                best_train_acc_avg=best_train_acc_avg,
                best_test_loss=best_test_loss,
                best_train_loss=best_train_loss,
                optimizer=optimizer.state_dict()
            )

            logger.append([epoch, optimizer.param_groups[0]['lr'],
                        train_out["loss"], train_out["acc_avg"], train_out["acc"],
                        test_out["loss"], test_out["acc_avg"], test_out["acc"]])

            # log results and save to csv.sa
            printf(f"Epoch [{epoch}] [Best test acc: {best_test_acc}%] \n")
            data = [epoch, round(train_out["loss"], 3), round(train_out["acc"], 2), 
                           round(test_out["loss"], 3), round(test_out['acc'], 2)]
            writer.writerow(data)
            csvfile.flush()
            gc.collect()
            torch.cuda.empty_cache()

        # save best resulte
        max_data = ['Best', round(best_train_loss, 3), round(best_train_acc, 2), 
                            round(best_test_loss, 2), round(best_test_acc, 2) ]
        writer.writerow(max_data) 
        csvfile.flush()
        logger.close()

        printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
        printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
        printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
        printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
        printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
        printf(f"++++++++" * 5)

def count_ncls_loss(x, label, criterion, random_number, mixup_p, lam, index):
    loss_list = []
    for outputs in x:
        # Mixup loss. 
        if random_number < mixup_p:
            loss = lam * criterion(outputs, label) + (1 - lam) * criterion(outputs, label[index])
        else:
            loss = criterion(outputs, label)
        loss_list.append(loss)
    loss_all = sum(loss_list) 
    return loss_all

def train_one_epoch(net, trainloader, optimizer, criterion, device, epoch, mixup, ncls_loss):
    if ncls_loss:
        print("===========Multi-CLS Loss==========\n")
    if mixup:
        print("===========mixup==========\n")

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    
    time_cost = datetime.datetime.now()
    trainloader = tqdm(trainloader, file = sys.stdout, ncols=110)
    trainloader.set_description(f"Epoch [{epoch}] [Train]")
    
    for step, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()

        mixup_p = mixup 
        if mixup:
            random_number = random.random()
        else:
            random_number = 1

        if random_number < mixup_p:  
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(data.size(0)).cuda()
            data = lam * data + (1 - lam) * data[index, :]
        else:
            lam = 0
            index = 0

        if not ncls_loss:
            outputs = net(data)
            if random_number < mixup_p:
                loss = lam * criterion(outputs, label) + (1 - lam) * criterion(outputs, label[index])
            else:
                loss = criterion(outputs, label)
        else:
            outputs_ = net(data)   
            loss = count_ncls_loss(outputs_, label, criterion, random_number, mixup_p, lam, index)
            outputs = outputs_[-1]  

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

        if random_number >= mixup_p:
            # calculate loss and accuracy, ouput log information
            preds = outputs.max(dim=1)[1]

            total += label.size(0)
            correct += preds.eq(label).sum().item()
            
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            trainloader.set_postfix(Loss=train_loss / (step + 1), Acc='%.3f%% (%d/%d)' % (100. * correct / total, correct, total))


    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
     
    return {
        "loss": float("%.3f" % (train_loss / (step + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }

def validate(net, testloader, criterion, device, epoch, ncls_loss):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    score_frag = []
    save_score=True

    time_cost = datetime.datetime.now()
    with torch.no_grad():
        test_dataset = testloader.dataset
        testloader = tqdm(testloader, file = sys.stdout, ncols=110)
        testloader.set_description(f"Epoch [{epoch}] [Test ]")
        for step, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            if not ncls_loss:
                outputs = net(data)
                loss = criterion(outputs, label)
            else:
                outputs_ = net(data)  
                loss = count_ncls_loss(outputs_, label, criterion, random_number=0, mixup_p=0, lam=0, index=0)
                outputs = outputs_[-1]  

            score_frag.append(outputs.data.cpu().numpy())

            test_loss += loss.item()

            preds = outputs.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            total += label.size(0)
            correct += preds.eq(label).sum().item()

            testloader.set_postfix(Loss=test_loss / (step + 1), Acc='%.3f%% (%d/%d)' % (100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    score = np.concatenate(score_frag)
    score_dict = dict(
                zip(test_dataset.sample_name, score))

    return {
        "loss": float("%.3f" % (test_loss / (step + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }

if __name__ == '__main__':
    main()