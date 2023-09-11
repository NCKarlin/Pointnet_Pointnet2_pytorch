'''
This is the utility function script for the PointNet++ implementation in all its 
variations.
We are mainly concerned with the semantic segmentation, based on the multi-scale grouping.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


# NORMALIZATION OF THE POINT CLOUD (in the Dataloader?)
def pc_normalize(pc):
    '''
    This function is used to 'normalize' the point cloud, so the model can learn location
    independent features (and does not somehow incorporate a specific bias towards certain 
    coordinates). 
    The normalization process (enclosing all points in a unit sphere) is as follows:
    1. Definition of the centroid or the entire PC.
    2. Shifting all points by subtracting the defined centroid
    3. Determining the noralizing parameter m (furthest distance)
    4. Normalizing by dividing the PC by m
    
    In this implementation the normalization of the point cloud is usually performed 
    within the dataloader, specifically in the _getitem_ function. 
    '''
    centroid = np.mean(pc, axis=0) # 1. 
    pc -= centroid  # 2. 
    furthest_distance = np.max(np.sqrt(np.sum(abs(pc**2, axis=1))))  # 3. 
    pc /=  furthest_distance # 4. 
    return pc

# CALCULATE EUCLID DISTANCE BETWEEN POINTS
def square_distance(src, dst):
    """
    This function returns the squared distance between source points and target points.
    Therefore it is used within the ball query function, where the squared distance
    is compared to the squared radius set for the local hood. 
    
    This is used for/ in:
    - determining the grouping of the ball query
    - PointNetFeaturePropagation

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
    Indexing points according to new group index.
    
    This is used for:
    - (re-) indexing points when they are being grouped
        - sample_and_group function -> grouping layer
    - PointNetSetAbstractionMsg
    - PointNetFeaturePropagation
    
    Input:
        points: input points data, [Batch/ Blocks, Num Points, Num Features]
        idx: sample index data, [Batch/ Blocks, New Num Points]
    Return:
        new_points: indexed points data, [Batches/ Blocks, New Num Points, Num Features]
    """
    device = points.device
    B = points.shape[0] # B is number of batches/ blocks
    
    # Old mechanism to check for the shape creation
    batch_shape = list(idx.shape)
    batch_shape[1:] = [1] * (len(list(idx.shape)) - 1)
    sample_shape = list(idx.shape)
    sample_shape[0] = 1
    
    # Creating batch indices and select corresponding points
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(batch_shape).repeat(sample_shape)
    new_points = points[batch_indices, idx, :]
    
    # Return reindexed points according to batch/ block index
    return new_points


# INDEXING OF POINTS 
def index_points_old(points, idx):
    """
    Indexing points according to new group index.
    
    This is used for:
    - (re-) indexing points when they are being grouped
        - sample_and_group function -> grouping layer
    - PointNetSetAbstractionMsg
    - PointNetFeaturePropagation
    
    Input:
        points: input points data, [Batch/ Blocks, Num Points, Num Features]
        idx: sample index data, [Batch/ Blocks, New Num Points]
    Return:
        new_points: indexed points data, [Batches/ Blocks, New Num Points, Num Features]
    """
    device = points.device
    B = points.shape[0] # B is number of batches/ blocks
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # Batch/ Block indexing for input points
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) #.to(device)
    new_points = points[batch_indices, idx, :]
    
    # Return reindexed points according to batch/ block index
    return new_points

# FPS FOR SAMPLING LAYER
def farthest_point_sample(xyz, npoint):
    """
    Theoretical Explanation:
    The farthest point sampling algo/ method is used in order to evenly cover the whole
    set, so the model can learn local features over the entire set, rather than just a 
    random or specific region. 
    
    This is used for:
    - for the sampling in the sample_and_group function
    - PointNetSetAbstractionMsg
    
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # Create "empty" tensor for number of samples
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # Setting of maximum distance?
    distance = torch.ones(B, N).to(device) * 1e10
    # Random sample indices from original pc
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # shape: (B,)
    # Create batch indices
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # Looping through number of wanted sample points
    for i in range(npoint):
        # Assigning random indices from original pc to centroids
        centroids[:, i] = farthest
        # TODO: what is being pulled from the original pc here?
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # Calculating the distance between original pc and pulled centroid?
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Masking for points below maximum distance
        mask = dist < distance
        # TODO: Is this really needed? Why not use dist[mask] directly?
        distance[mask] = dist[mask]
        # Selecting the farthest distance
        farthest = torch.max(distance, -1)[1]
    # Returning sampled centroid point of input pc 
    return centroids


# GROUPING/ INDEXING POINTS ACCORDING TO QUERY BALL
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Theoretical Explanation:
    This method finds all the points that are within a radius to the query point(s).
    The amount of sample points within the local neighbourhood is bound by an upper limit.
    It guarantees a fixed region scale, so local region features are more generalizable
    across space, which is essential for the local pattern recognition for semantic 
    labelling.
    
    This is used for/ in:
    - sample_and_group function
        -> more specifically the Grouping Layer
    - PointNetSetAbstractionMsg
    
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
    # Determining the distance between query points and PC
    sqrdists = square_distance(new_xyz, xyz)
    # All points whose squared distance is greater than the squared radius are assigned N 
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    # Returning the indexed & grouped points
    return group_idx


#! ONLY USED IN THE SINGLE SCALE SAMPLING SET ABSTRACTION
# SAMPLING AND GROUPING LAYER
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Theoretical Explanation:
    The Sampling and Grouping layer in the model are used in order to sample evenly
    across the entire set - giving centroids around which to group the local region
    points - and subsequently group the points within a given local region, so that
    the local context can be learned.
    The operations for the different layers are as follows:
    1. Sampling Layer:
        1.1 Input:
        -> input points, which basically means the PC
        1.2 Output:
        -> number of sample points, which ensure good coverage of entire set through FPS
    2. Grouping Layer:
        2.1 Input:
        -> point set of size Nx(d+C) 
            - N: number of points in pc
            - d: dimensions of pc (3D)
            - C: point features
        -> coordinates of centroids of size N'xd 
            - N: number of centroid points
            - d: dimensions of centroid points (3D)
        2.2 Output:
        -> Groups of point sets of size N'xKx(d+C)
            - N': number of centroid points
            - K: number of points in the neighbourhood of the centroids
            - d: dimensions of those points (3D)
            - C: point features
    
    Input:
        npoint: number of centroids to sample around
        radius: local region radius
        nsample: max sample number in local region
        xyz: input points position data, [B, N, 3]
            - B: batch/ block size
            - N: number of points within that batch/ block
            - 3: xyz-coordinates of points
        points: input points data, [B, N, D]
            - B: batch/ block size
            - N: number of points within that batch/ block
            - D: dimensions + features of input pc
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # Sampling Layer
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    # Grouping Layer
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


#! ONLY NEEDED IF YOU DO SINGLE SCALE GROUPING
# Sampling and grouping with single scale instead of multiscale
def sample_and_group_all(xyz, points):
    """
    TODO: Come back and investigate why this is useful
    
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


#! NOT NEEDED IF USING MULTI-GROUP SCALING 
# SET ABSTRACTION: Sampling + Grouping + pointnet layer
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


# SET ABSTRACTION WITH MULTI-SCALE GROUPING: Sampling + Grouping + Pointnet Layer
class PointNetSetAbstractionMsg(nn.Module):
    #TODO: Double-check whether the procedures here make sense and the functions they use
    '''
    The PointnetSetAbstractionMSG class comprises all the operations for one set
    abstraction.
    This means it does the pass of a PC through the sampling, grouping and finally 
    the pointnet layer. 
    After one pass through the set abstraction, the model will have one (more) level
    of abstraction of the entire point cloud, and also their local regional features. 
    It is defined by the following parameters:
    - npoint: number of sampled centroids from FPS (int)
    - radius_list: list of different radii to define the ball query for sampling around
      centroids
        -> typically the radius list will increase for each abstraction for the MSG
    - nsample_list: list of maximum samples to be collected around each centroid within 
      the ball query
    - in_channel: number of inputs for set abstraction layer
        -> typically this value will increase with each set abstraction
    - mlp_list: List of the MLPs with their number of respective layer in- and outputs
    '''
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        #TODO: check the MLP creation accordingly
        # Module list of convolutional blocks with their respective layers
        self.conv_blocks = nn.ModuleList()
        # Module list of batch normalization layers after each convolutional layer
        self.bn_blocks = nn.ModuleList()
        # Looping trough number of MLP's
        for i in range(len(mlp_list)):
            # Creating Module lists for each MLP
            convs = nn.ModuleList()
            # Creating Module lists for each batch norm layer
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            # Looping through the layers of the respective MLP
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
            
        B, N, C = xyz.shape
        S = self.npoint
        # Determining and indexing the sampled query points
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        # Looping through the number of radii given in the radii list for MSG
        for i, radius in enumerate(self.radius_list):
            # Setting the upper limit for the number of points within ball query
            K = self.nsample_list[i]
            # Determining the point groups for each ball query
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # Reindexing the points according to the grouping
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            # if a input pc cloud was given
            if points is not None:
                # Re-index the points according to the group indexing from the respective radius
                grouped_points = index_points(points, group_idx)
                # Concatenation of the different grouped points 
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                # Else the grouped points are just the grouped locations
                grouped_points = grouped_xyz

            # Looping through the layers of the set abstraction MLP
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            
            # Max Pooling?
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # Appending the result of the max pooling to the new points?
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # Concatenate the new_points_list with waht??
        new_points_concat = torch.cat(new_points_list, dim=1)
        
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    '''
    Generally, the feature propagation in the PointNett ++ is done via hierarchical 
    propagation with distance-based interpolation and across level skip links. 
    This means, that for the feature propagation the point features from different 
    abstraction levels are propagated, and in the case of not matching abstraction
    dimensions, the "missing" point features are interpolated, based on their relative
    distance. The across level skip-links refer to the concatenation of the point data
    from the different abstraction levels (e.g.: level 1 and interpolated level 2).
    The interpolation occurs, because at the different abstraction levels, the point
    set size differs, according to the centroids chosen for each abstraction layer.
    '''
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: sampled input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # INTERPOLATION OF POINTS BETWEEN SET ABSTRACTION LAYERS
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # interpolation of points if feature levels are not equal
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # Creating distance-based weight for interpolation
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # Interpolation based on the distance weights
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        
        # Check for last feature propagation layer
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            # for last feature propagation layer -> points1 = None
            new_points = interpolated_points

        # new_points -> concatenated points1 with interpolated points
        new_points = new_points.permute(0, 2, 1)
        # Looping through MLP layers
        for i, conv in enumerate(self.mlp_convs):
            # Pull corresponding batch normalization layer
            bn = self.mlp_bns[i]
            # Convolve, Normalize, Activate
            new_points = F.relu(bn(conv(new_points)))
        return new_points

