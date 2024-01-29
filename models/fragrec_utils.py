import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


# LAYER CLASSES FOR MODEL CREATION
#########################################################################################

# SET ABSTRACTION WITH MULTI-SCALE GROUPING: Sampling + Grouping + Pointnet Layer
class PointNetSetAbstractionMsg(nn.Module):
    '''
    THEORY
    The PointnetSetAbstractionMSG class comprises all the operations for one set
    abstraction.
    This means it does the pass of a PC through the sampling, grouping and finally 
    the pointnet layer. 
    After one pass through the set abstraction, the model will have one (more) level
    of abstraction of the entire point cloud, and also their local features. 
    
    __Init__:
    - npoint:       number of sampled centroids from FPS (int)
    - radius_list:  list of different radii to define the ball query for sampling around
                    centroids
                     -> typically the radius list will increase for each abstraction for 
                        the MSG
    - nsample_list: list of maximum samples to be collected around each centroid within 
                    the ball query
    - in_channel:   number of inputs for set abstraction layer
                    -> typically this value will increase with each set abstraction
    - mlp_list:     List of the MLPs with their number of respective layer in- & outputs
    '''
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList() #module list for convolutional blocks
        self.bn_blocks = nn.ModuleList() #module list for batch normalization
        
        # Looping trough number of MLP's
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
        
            # Looping through the layers of the respective MLP
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            
            # Appending respective module lists to overall module list
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    # FORWARD PASS THROUGH SET ABSRACTION LAYER
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
            
        Shape clarifications:
        B:  batch size
        C:  xyz-coordinates [3]
        N:  number of points (per batch)
        D:  number of features for input points
        S:  subset of the N points
        D': increased number of features for output points from concatenation
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        
        # Determining and indexing the sampled query points
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        
        # Looping through the number of radii given in the radii list for MSG
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            # Setting the upper limit for the number of points within ball query
            K = self.nsample_list[i]
            # Determining the point groups for each ball query
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # Reindexing the points according to the grouping
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            
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
            
            # Max Pooling
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # Appending the result of the max pooling to the new points?
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        #TODO: check what is concatenated here
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
        #TODO: hceck dimensions before they are permuted 
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        #TODO: check the correctness of the variables concerning the shape

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


# BASIC UTILITY FUNCTIONS FOR MODEL LAYERS
#########################################################################################
# RE-INDEXING POINTS ACCORDING TO GROUPED
def index_points(points, idx):
    """
    Indexing points according to new group index.
    
    This is used for:
    - (re-) indexing points when they are being grouped
    
    Input:
        points: input points data, [Batch, Num Points, Num Features]
        idx: sample index data, [Batch, New Num Points]
    Return:
        new_points: indexed points data, [Batches, New Num Points, Num Features]
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
    
    # Return reindexed points according to batch index
    return new_points


# FPS FOR SAMPLING LAYER
def farthest_point_sample(xyz, npoint):
    """
    Theoretical Explanation:
    The farthest point sampling algo/ method is used in order to evenly cover the whole
    set, so the model can learn local features over the entire set, rather than just a 
    random or specific region. 

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


# CALCULATE EUCLID DISTANCE BETWEEN POINTS
def square_distance(src, dst):
    """
    This function returns the squared distance between source points and target points.
    Therefore it is used within the ball query function, where the squared distance
    is compared to the squared radius set for the local hood. 

    src^T * dst = xn * xm + yn * ym + zn * zm;
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