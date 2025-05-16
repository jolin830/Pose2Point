def sample_joint_amplitude(Joints_amplitude, npoint):
    """
    Input:
        Joints_amplitude: pointcloud data, [B, 25]
        npoint: number of samples
    Return:
        Joints_sample: samples of each joints [B, 25]
    """
    B, num_joints = Joints_amplitude.shape
    
    # 归一化得到概率分布
    probs = Joints_amplitude / Joints_amplitude.sum(dim=1, keepdim=True)
    
    # 初始化采样结果
    Joints_sample = torch.zeros_like(Joints_amplitude, dtype=torch.int32)
    
    for i in range(B):
        # 依据概率分布进行采样
        sampled_indices = torch.multinomial(probs[i], npoint, replacement=True)
        
        # 使用 bincount 统计采样次数
        Joints_sample[i] = torch.bincount(sampled_indices, minlength=num_joints)
    
    return Joints_sample


def uniform_joints_sampling(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 5]
        npoint: number of samples
        farthest_point_sample: function to perform farthest point sampling
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, _ = xyz.shape
    num_joints = 25
    points_per_joint = npoint // num_joints  # 每个关节点应采样的点数
    remaining_points = npoint % num_joints  # 余下的点数
    key_joints = [24, 22, 4, 20, 16, 
                  25, 23, 3, 19, 15, 
                  12, 8, 9, 21, 5, 
                  18, 14, 11, 7, 17, 
                  13, 10, 6, 2, 1]
    
    centroids = torch.zeros((B, npoint), dtype=torch.long).to(device)
    
    sampled_indices = []
    tolerance = 1e-6

    for joint_id in range(1, num_joints + 1):

        # 获取属于当前关节点的所有点的原始索引，形状为 [B, joint_n]
        # joint_indices = (xyz[:, :, 4] == (joint_id/25-0.5)).nonzero(as_tuple=True)[1].view(B, -1)
        # joint_indices = (torch.abs(xyz[:, :, 4] - (joint_id/25-0.5)) < tolerance).nonzero(as_tuple=True)[1].view(B, -1)
        joint_indices = (torch.abs(xyz[:, :, 4] - (joint_id * 0.001)) < tolerance).nonzero(as_tuple=True)[1].view(B, -1)
        
        # 获取当前关节点的点云坐标，形状为 [B, joint_n, 3]
        joint_points = index_points(xyz, joint_indices)
    
        # 使用最远点采样选择 points_per_joint 个点的局部索引
        # fps_per_joint = points_per_joint + 1 if joint_id <= remaining_points else points_per_joint
        fps_per_joint = points_per_joint + 1 if joint_id in key_joints[:remaining_points] else points_per_joint
        local_sampled_joint_indices = farthest_point_sample(joint_points, fps_per_joint)

        # 将局部索引映射为全局索引
        global_sampled_joint_indices = joint_indices.gather(1, local_sampled_joint_indices)

        sampled_indices.append(global_sampled_joint_indices)

    # 将所有关节点的索引合并
    centroids = torch.cat(sampled_indices, dim=1)

    return centroids



def farthest_point_sample2(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    device = xyz.device
    N = xyz.shape[0]
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)
    distances = torch.full((N,), float('inf')).to(device)
    farthest = torch.randint(0, N, (1,)).to(device)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]  # [1, 3]
        dist = torch.sum((xyz - centroid) ** 2, dim=1)  # [N]
        distances = torch.min(distances, dist)
        farthest = torch.argmax(distances)
    
    return centroids

def uniform_amplitude_sampling(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 5]
        npoint: number of samples
        farthest_point_sample: function to perform farthest point sampling
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    num_joints = 25
    centroids = torch.zeros((B, npoint), dtype=torch.long).to(device)

    for b in range(B):
        sampled_indices = []
        xyz_b = xyz[b]

        # 1. 计算该样本每个关节点的动作幅度 → 关节点采样个数
        tmp = xyz_b.view(64, 2, 25, 5)

        diff1 = tmp[1:,0, :, :3] - tmp[:-1, 0, :, :3] 
        dist = torch.norm(diff1, dim=-1)    # [B, frames-1]
        amplitude1 = torch.sum(dist, dim=0)  # [3]

        diff2 = tmp[1:,1, :, :3] - tmp[:-1, 1, :, :3]
        dist = torch.norm(diff2, dim=-1)    # [B, frames-1]
        amplitude2 = torch.sum(dist, dim=0)

        amplitude = amplitude1 + amplitude2  # [25]
        
        # 2. 根据每个关节点的动作幅度进行动态采样
        proba = amplitude / amplitude.sum()
        sampled_indices_tmp = torch.multinomial(proba, npoint, replacement=True)
        fps_each_joints = torch.bincount(sampled_indices_tmp, minlength=num_joints)

    
        # 动作幅度 → 关节点采样个数
        for joint_id in range(1, num_joints + 1):

            # 获取属于当前关节点的所有点的原始索引，形状为 [joint_n]
            joint_indices = (torch.abs(xyz_b[:,4] - (joint_id * 0.001)) < 1e-6).nonzero(as_tuple=True)[0]
            joint_points = xyz_b[joint_indices]
            
            """根据每个关节点的动作态势进行动态采样"""
            local_sampled_joint_indices = farthest_point_sample2(joint_points, fps_each_joints[joint_id-1])
            
            # 将局部索引映射为全局索引
            global_sampled_joint_indices = joint_indices[local_sampled_joint_indices]
            sampled_indices.append(global_sampled_joint_indices)
        
        # 将所有关节点的索引合并
        centroids[b] = torch.cat(sampled_indices, dim=0)

    return centroids




