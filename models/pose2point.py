
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from einops import rearrange
import numpy as np

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def square_distance_out(src, dst, large_value=100):
    """
    Calculate Euclid distance between each two points.
    If the last channel (C-1) of a point from src and dst is the same, 
    set the distance between them to a very large value.
    
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
        large_value: the value to set for the distance if the last channel is the same.
        
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    
    # Calculate the initial squared distance
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    # Check if the last channel (C-1) of src and dst are the same
    last_channel_src = src[:, :, -1].unsqueeze(-1)                   # [B, N, 1]
    last_channel_dst = dst[:, :, -1].unsqueeze(-1).permute(0, 2, 1)  # [B, 1, M]
    
    # Set distance to a very large value where the last channel is the same
    mask = (last_channel_src == last_channel_dst)  # [B, N, M], True if channels match
    dist[mask] += large_value  # Apply the large value to matching points
    
    return dist
    
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 5)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):#
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape 
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True,  sample="fast",
                normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.sample = sample
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=5 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def uniform_joints_sampling(self,xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 5]
            npoint: number of samples
            farthest_point_sample: function to perform farthest point sampling
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        B, N, _ = xyz.shape
        device = xyz.device
        num_joints = 25
        points_per_joint = npoint // num_joints 
        remaining_points = npoint % num_joints 
        # key_joints = [24, 22, 4, 20, 16, 
        #             25, 23, 3, 19, 15, 
        #             12, 8, 9, 21, 5, 
        #             18, 14, 11, 7, 17, 
        #             13, 10, 6, 2, 1]
        key_joints = [23, 21, 3, 19, 15, 
                    24, 22, 2, 18, 14, 
                    11, 7, 8, 20, 4, 
                    17, 13, 10, 6, 16, 
                    12, 9, 5, 1, 0]
        
        centroids = torch.zeros((B, npoint), dtype=torch.long).to(device)
        
        sampled_indices = []
        tolerance = 1e-6
        # (0~24 / 24) - 0.5
        for joint_id in range(num_joints):

            # joint_indices = (xyz[:, :, 4] == (joint_id/25-0.5)).nonzero(as_tuple=True)[1].view(B, -1)
            joint_indices = (torch.abs(xyz[:, :, 4] - (joint_id/(num_joints-1)-0.5)) < tolerance).nonzero(as_tuple=True)[1].view(B, -1)
            # joint_indices = (torch.abs(xyz[:, :, 4] - (joint_id * 0.001)) < tolerance).nonzero(as_tuple=True)[1].view(B, -1)
            
            joint_points = index_points(xyz, joint_indices) # [B, joint_n, 3]
        
            # 使用最远点采样选择 points_per_joint 个点的局部索引
            # fps_per_joint = points_per_joint + 1 if joint_id <= remaining_points else points_per_joint
            fps_per_joint = points_per_joint + 1 if joint_id in key_joints[:remaining_points] else points_per_joint

            local_sampled_joint_indices = farthest_point_sample(joint_points, fps_per_joint)  # slow
            # local_sampled_joint_indices = pointnet2_utils.furthest_point_sample(joint_points.float(), fps_per_joint).long() # fast
            
            global_sampled_joint_indices = joint_indices.gather(1, local_sampled_joint_indices)

            sampled_indices.append(global_sampled_joint_indices)

        centroids = torch.cat(sampled_indices, dim=1)

        return centroids

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz] [1, 16000, 5]

        # ========= FPS =========
        if self.sample=="slow":
            fps_idx = farthest_point_sample(xyz, self.groups).long() 
        elif self.sample=="fast":
            fps_idx = pointnet2_utils.furthest_point_sample(xyz.float(), self.groups).long()  
        elif self.sample=="uniform_joints":
            fps_idx = self.uniform_joints_sampling(xyz.float(), self.groups).long()
        elif self.sample=="resample":
            fps_idx = self.point_resample(xyz.float(), self.groups)

        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        # ========= KNN =========
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta 

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1) 
        return new_xyz, new_points

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)   # ConvBNReLURes1D
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )   
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)

class Classifier_Block(nn.Module):
    def __init__(self, out_channel, class_num, act):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(Classifier_Block, self).__init__()
        self.class_num = class_num
        self.act = act
        if out_channel<512:
            self.classifier = nn.Sequential(
                        nn.Linear(out_channel, 128),   
                        nn.BatchNorm1d(128),
                        self.act,
                        nn.Dropout(0.5),
                        nn.Linear(128, self.class_num)
                    )
        elif out_channel<1024:
            self.classifier = nn.Sequential(
                        nn.Linear(out_channel, out_channel//2),
                        nn.BatchNorm1d(out_channel//2),
                        self.act,
                        nn.Dropout(0.5),
                        nn.Linear( out_channel//2, self.class_num)
                    )
        else:
            self.classifier = nn.Sequential(
                                nn.Linear(out_channel, out_channel//2),
                                nn.BatchNorm1d(out_channel//2),
                                self.act,
                                nn.Dropout(0.5),
                                nn.Linear(out_channel//2, out_channel//4),
                                nn.BatchNorm1d(out_channel//4),
                                self.act,
                                nn.Dropout(0.5),
                                nn.Linear( out_channel//4, self.class_num)
                            )

    def forward(self, x):  # [b, d, g]
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x

class ContextCluster(nn.Module):
    def __init__(self, dim, heads=4, head_dim=24):
        super(ContextCluster, self).__init__()
        self.heads = heads
        self.head_dim=head_dim
        self.fc1 = nn.Linear(dim, heads*head_dim)
        self.fc2 = nn.Linear(heads*head_dim, dim)
        self.fc_v = nn.Linear(dim, heads*head_dim)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

    def forward(self, x): 
        res = x
        x = rearrange(x, "b d k -> b k d") 
        value = self.fc_v(x)  # [b,k,head*head_d]
        x = self.fc1(x) # [b,k,head*head_d]
        x = rearrange(x, "b k (h d) -> (b h) k d", h=self.heads)  # [b,k,d]
        value = rearrange(value, "b k (h d) -> (b h) k d", h=self.heads)  # [b,k,d]
        center = x.mean(dim=1, keepdim=True)  # [b,1,d]
        value_center = value.mean(dim=1, keepdim=True)  # [b,1,d]
        sim = nn.ReLU(self.sim_beta + self.sim_alpha * pairwise_cos_sim(center, x) )#[B,1,k]
        out = ( (value.unsqueeze(dim=1)*sim.unsqueeze(dim=-1) ).sum(dim=2) + value_center)/ (sim.sum(dim=-1,keepdim=True)+ 1.0) # [B,M,D]
        out = out*(sim.squeeze(dim=1).unsqueeze(dim=-1)) # [b,k,d]
        out = rearrange(out, "(b h) k d -> b k (h d)", h=self.heads)  # [b,k,d]
        out = self.fc2(out)
        out = rearrange(out, "b k d -> b d k")
        return res + out 

class Model(nn.Module):
    def __init__(self, points=5000, class_num=120, embed_dim=64, groups=1, res_expansion=1.0,sample="fast",
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(5, embed_dim, bias=bias, activation=activation)#0614 c=3->c=5
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        self.classifier_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        self.act = get_activation(activation)

        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, sample, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)
            classifier = Classifier_Block(out_channel, class_num, self.act)
            self.classifier_list.append(classifier)

            last_channel = out_channel
        
    def forward_embeddings(self, x):
        b, c, img_v, img_t, m = x.size()  
        range_v = torch.arange(0, img_v, step=1).to("cuda") / (img_v - 1.0) 
        range_t = torch.arange(0, img_t, step=1).to("cuda") / (img_t - 1.0) 
        fea_pos = torch.stack(torch.meshgrid(range_v, range_t, indexing='ij'), dim=-1).float().to("cuda")
        fea_pos = fea_pos
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(b*m, -1, -1, -1).to("cuda") 
        
        x = x.permute(0, 4, 1, 2, 3).reshape(b*m, c, img_v, img_t).to("cuda")
        x = torch.cat([x[torch.arange(b*m),:,:,:], pos], dim=1).to("cuda") 
        x = x.reshape(b, m, c+2, img_v, img_t).permute(0,2,3,4,1)  
        return x


    def trans_2_pc(self, x, n, c):
        x = x.reshape(n, c+2, -1).float()  
        xyz = x.permute(0, 2, 1).float()     
        x = self.embedding(x)  
        return xyz, x

    def forward(self, x):
        n, c, T, V, M = x.size()
        x = self.forward_embeddings(x) # -> [B, C+2, T, V, M]
        xyz, x = self.trans_2_pc(x, n, c)
        x_all = []

        # ============== backbone ===============
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g] 
            x = self.pos_blocks_list[i](x)  # [b,d,g] 
            if i>0: 
                x_ = self.classifier_list[i](x)
                x_all.append(x_)

        return x_all


def Pose2Point(num_classes=60, num_points=4096, sample="fast",**kwargs) -> Model:
    return Model(points=num_points, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0, 
                   sample=sample, 
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

def Pose2Point_5(num_classes=60, num_points=1024,**kwargs) -> Model:    # 5 layers
    return Model(points=num_points, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,   
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2, 2], pre_blocks=[2, 2, 2, 2, 2], pos_blocks=[2, 2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24, 24], reducers=[2, 2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    data = torch.rand(1,3,64,25,2).cuda()
    print("===> testing Pose2Point ...")
    model = Pose2Point(sample="slow").cuda()
    out = model(data)
    print(out)
