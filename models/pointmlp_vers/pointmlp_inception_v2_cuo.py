
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from einops import rearrange
import numpy as np

def get_activation(activation):#选择激活函数
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
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 5)##############################################################
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
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
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
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=5 if self.use_xyz else 0#ybq0615 3->5
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def point_resample(self,points,npoints,stat):
        num_curr_pts = points.shape[1]
        if stat == "train":
            if num_curr_pts > npoints:  # point resampling strategy
                if npoints == 128:
                    point_all = 150
                elif npoints == 256:
                    point_all = 300
                elif npoints == 512:
                    point_all = 600
                elif npoints == 1024:
                    point_all = 1200
                elif npoints == 2048:
                    point_all = 2400
                elif npoints == 4096:
                    point_all = 4800
                else:
                    raise NotImplementedError()

                if  num_curr_pts < point_all:
                    point_all = num_curr_pts
        else:
            point_all = npoints

        # fps_idx = farthest_point_sample(points, point_all).long()  # 慢但效果好一点
        fps_idx = pointnet2_utils.furthest_point_sample(points.float(), point_all).long()  # 快但效果差一点
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]

        return fps_idx

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz] [1, 16000, 5]
        # fps_idx = farthest_point_sample(xyz, self.groups).long()  # 慢但效果好一点
        fps_idx = pointnet2_utils.furthest_point_sample(xyz.float(), self.groups).long()  # 快但效果差一点

        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

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
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta # 消融

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

class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4, activation):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.act = get_activation(activation)
        self.p1_1 = nn.Conv1d(in_c, c1, kernel_size=1)  # conv2d->Conv1d
        self.n1_1 = nn.BatchNorm1d(c1)  # add BatchNorm1d, 防止溢出

        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv1d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv1d(c2[0], c2[1], kernel_size=3, padding=1)
        self.n2_1 = nn.BatchNorm1d(c2[1])  # add BatchNorm1d

        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv1d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv1d(c3[0], c3[1], kernel_size=5, padding=2)
        self.n3_1 = nn.BatchNorm1d(c3[1])  # add BatchNorm1d

        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv1d(in_c, c4, kernel_size=1)
        self.n4_1 = nn.BatchNorm1d(c4)  # add BatchNorm1d


    def forward(self, x):
        p1 = self.act(self.n1_1(self.p1_1(x)))
        p2 = self.act(self.n2_1(self.p2_2(self.act(self.p2_1(x)))))
        p3 = self.act(self.n3_1(self.p3_2(self.act(self.p3_1(x)))))
        p4 = self.act(self.n4_1(self.p4_2(self.p4_1(x))))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出

        #------- wo bn---------
        # p1 = self.act((self.p1_1(x)))
        # p2 = self.act((self.p2_2(self.act(self.p2_1(x)))))
        # p3 = self.act((self.p3_2(self.act(self.p3_1(x)))))
        # p4 = self.act((self.p4_2(self.p4_1(x))))
        # return p1+p2+p3+p4  # 在通道维上点加

class ConvBNReLURes1D_inception(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D_inception, self).__init__()

        self.act = get_activation(activation)
        channel_res = int(channel * res_expansion)
        self.inception1 =  Inception(in_c=channel, c1=channel_res//4, c2=(channel_res//2, channel_res//4), c3=(channel_res//2, channel_res//4), c4=channel_res//4, activation=activation)
        self.inception2 = Inception(in_c=channel_res, c1=channel//4, c2=(channel//2, channel//4), c3=(channel//2, channel//4), c4=channel//4, activation=activation)
        # self.inception1 =  Inception(in_c=channel, c1=channel_res, c2=(channel_res//2, channel_res), c3=(channel_res//2, channel_res), c4=channel_res,activation=activation)
        # self.inception2 = Inception(in_c=channel_res, c1=channel, c2=(channel//2, channel), c3=(channel//2, channel), c4=channel, activation=activation)
        self.net1 = nn.Sequential(
            self.inception1,
            nn.BatchNorm1d(channel_res),
            self.act
        )

        self.net2 = nn.Sequential(
            self.inception2,
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
                ConvBNReLURes1D_inception(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)   # ConvBNReLURes1D
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        # print("----------pintmlp shape="+str(x.size())+"----------")
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
                ConvBNReLURes1D_inception(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )   # ConvBNReLURes1D
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)

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

    def forward(self, x): #[b,d,s]中间是通道 最后是点数
        res = x
        x = rearrange(x, "b d k -> b k d")  # [B, 2048, 128]
        value = self.fc_v(x)  # [b,k,head*head_d]
        x = self.fc1(x) # [b,k,head*head_d]
        x = rearrange(x, "b k (h d) -> (b h) k d", h=self.heads)  # [b,k,d]
        value = rearrange(value, "b k (h d) -> (b h) k d", h=self.heads)  # [b,k,d]
        center = x.mean(dim=1, keepdim=True)  # [b,1,d]
        value_center = value.mean(dim=1, keepdim=True)  # [b,1,d]
        sim = nn.ReLU(self.sim_beta + self.sim_alpha * pairwise_cos_sim(center, x) )#[B,1,k]
        # out [b, 1, d]
        out = ( (value.unsqueeze(dim=1)*sim.unsqueeze(dim=-1) ).sum(dim=2) + value_center)/ (sim.sum(dim=-1,keepdim=True)+ 1.0) # [B,M,D]
        out = out*(sim.squeeze(dim=1).unsqueeze(dim=-1)) # [b,k,d]
        out = rearrange(out, "(b h) k d -> b k (h d)", h=self.heads)  # [b,k,d]
        out = self.fc2(out)
        out = rearrange(out, "b k d -> b d k")
        return res + out    # 残差链接 点加


class Model(nn.Module):
    def __init__(self, points=5000, class_num=120, embed_dim=64, groups=1, res_expansion=1.0,
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
        # self.cocs_list =  nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points

        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
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
            # cocs_block = ContextCluster(out_channel)
            # self.cocs_list.append(cocs_block)

            last_channel = out_channel

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

        
    def forward_embeddings(self, x):
        b, c, img_v, img_t, m = x.size()  # [15, 3, 64, 25, 2]
        range_v = torch.arange(0, img_v, step=1).to("cuda") / (img_v - 1.0) # torch.Size([64])
        range_t = torch.arange(0, img_t, step=1).to("cuda") / (img_t - 1.0) # torch.Size([25])
        fea_pos = torch.stack(torch.meshgrid(range_v, range_t, indexing='ij'), dim=-1).float().to("cuda") # torch.Size([64, 25, 2])
        fea_pos = fea_pos
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(b*m, -1, -1, -1).to("cuda") #[8, 2, 64, 25]
        
        # [b, c, T, V, M] -> [b, M, c, T, V] -> [b*M, c, T, V]
        x = x.permute(0, 4, 1, 2, 3).reshape(b*m, c, img_v, img_t).to("cuda")
        x = torch.cat([x[torch.arange(b*m),:,:,:], pos], dim=1).to("cuda")#for循环优化
        x = x.reshape(b, m, c+2, img_v, img_t).permute(0,2,3,4,1) # [16, 5, 64, 25] -> [8, 2, 5, 64, 25] -> [8, 5, 64, 25, 2]
        return x
    
    def forward(self, x):
        n, c, T, V, M = x.size()
        x = self.forward_embeddings(x) # -> [B, C+2, T, V, M]
        x = x.reshape(n, c+2, -1) 
        xyz = x.permute(0, 2, 1)       # -> [B, 16000, c+2]
        x=x.float()
        x = self.embedding(x)          # -> [B, 64, 16000]

        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d] xyz:[2, 512, 5]
            x = self.pre_blocks_list[i](x)  # [b,d,g] [2, 128, 512]
            x = self.pos_blocks_list[i](x)  # [b,d,g] [2, 128, 512]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


# def pointMLP(num_classes=60, **kwargs) -> Model:
def pointMLP(num_classes=60, num_points=4096,**kwargs) -> Model:
    return Model(points=num_points, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,     # points=1024--*5-->5120 2的倍数 4096
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> Model:
    return Model(points=4096, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,    # points=1024--*5-->5120  2的倍数 4096
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                   k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2], **kwargs)

if __name__ == '__main__':
    data = torch.rand(1, 3, 64, 25, 2).cuda()
    print("===> testing pointMLP ...")
    model = pointMLP().cuda()
    out = model(data)
    print(out.shape)