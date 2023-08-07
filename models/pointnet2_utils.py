'''
This is the utility function script for the PointNet++ implementation in all its 
variations.
We are mainly concerned with the semantic segmentation, (based on the multi-scale grouping.)
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
    3. Determining the normalizing parameter m (furthest distance)
    4. Normalizing by dividing the PC by m
    
    In this implementation the normalization of the point cloud is usually performed 
    within the dataloader, specifically in the _getitem_ function. 
    '''
    # what is l used for?? TODO Delete?
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0) # 1. 
    pc = pc - centroid  # 2. 
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 3. 
    pc = pc / m # 4. 
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
    #? why multiplying -2? should we maybe integrate "regular" distance calculation?
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# INDEXING OF POINTS 
def index_points(points, idx):
    """
    Indexing points according to new group index.
    
    This is used for:
    - (re-) indexing points when they are being grouped
        - sample_and_group function -> grouping layer
    - PointNetSetAbstractionMsg
    - PointNetFeaturePropagation
    

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0] # B is number of blocks
    view_shape = list(idx.shape) # [8, 1024, 32]
    view_shape[1:] = [1] * (len(view_shape) - 1) # [8, 1, 1]
    repeat_shape = list(idx.shape) # [8, 1024, 32]
    repeat_shape[0] = 1 # [1, 1024, 32]
    # Batch/ Block indexing for input points
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) #[8, 1024, 32]
    # Selecting correctly batch and group indexed points from input PC
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


# GROUPING AND (RE-) INDEXING POINTS ACCORDING TO QUERY BALL
def query_ball_point(radius, nsample, xyz, new_xyz):
    """ THEORY
    This method finds all the points that are within a radius to the query point(s).
    The amount of sample points within the local neighbourhood is bound by an upper limit.
    It guarantees a fixed region scale, so local region features are more generalizable
    across space, which is essential for the local pattern recognition for semantic 
    labelling.

    INPUT:
        radius: local region radius -> float
        nsample: max sample number in local region -> int
        xyz: all/input points -> [num_blocks, num_input_points, 3D coord]
        new_xyz: query points -> [num_blocks, num_centroids, 3D coord]
    
    OUTPUT:
        group_idx: grouped points index -> [num_blocks, num_centroids, nsample] [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # Creating group indexed tensor
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # -> [num_blocks, num_centroids, num_input_point]
    # Determining the distance between query points and PC
    sqrdists = square_distance(new_xyz, xyz)
    # -> [num_blocks, num_centroids, num_input_point]
    # Set all points values, whose distance is greater than the radius to 4096
    group_idx[sqrdists > radius ** 2] = N #? is it squared because of squared distances?
    #TODO: Think about maybe doing a little more random selection among the closest points?
    # Sorting along last dimension (points per centroid), and taking the nsample closest ones
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # [num_blocks, num_centroids, nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # -> [num_blocks, num_centroids, nsample]
    mask = group_idx == N
    # -> masking the group_idx tensor with value 4096 -> 0: index of points within radius
    group_idx[mask] = group_first[mask]
    # Replacing the indices outside of radius with index of first/ closest point
    
    # Returning the indexed & grouped points
    return group_idx


#! ONLY USED IN THE SINGLE SCALE SAMPLING SET ABSTRACTION
# SAMPLING AND GROUPING LAYER
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """ THEORY
    The Sampling and Grouping layer in the model are used in order to sample evenly
    across the entire set - giving centroids around which to group the local region
    points - and subsequently group the points within a given local region, so that
    the local context can be learned.
    The operations for the different layers are as follows:
    
    INPUT:
        npoint: number of centroids to sample around -> int
        radius: local region radius -> float
        nsample: max sample number in local region -> int
        xyz: input points position data [B, N, 3]
            -> [num_blocks, num_points/ num_centroids, 3D coord]
        points: input points data/ points to be sampled from [B, N, D]
            -> [num_blocks, num_points/ num_centroids, features] 
            
    OUTPUT:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape #-> xyz [num_blocks, num_points, 3D coordinates]
    S = npoint #-> number of centroids to be sampled around
    
    
    ''' SAMPLING "LAYER"
    Within the sampling layer the respective centroids are selected and subsequently
    sampled around, meaning a certain number of points within a given radius around
    the selected centroids are picked.
     
    INPUT:
        xyz: coordinates of the points to be sampled
            -> [num_blocks, num_points, 3D coordinates]
        npoint: number of centroids to be sampled around
            -> int
    
    OUTPUT:
        new_xyz: coordinates of the centroids to be sampled around
            -> [num_blocks, npoint, 3D coordinates]
    '''
    fps_idx = farthest_point_sample(xyz, npoint) # -> [num_blocks, num_centroids]
    new_xyz = index_points(xyz, fps_idx) # -> [num_blocks, npoint, 3D coordinates]
    
    
    ''' GROUPING LAYER
    Within the grouping layer, the points around the centroids are sampled and grouped
    accordingly to arrive at a grouped representations of local centroid neighborhoods
    of the input PC. 
    
    INPUT:
        radius: radius within which the points -> float
        nsample: number of points to sample around each centroid -> int
        xyz: coordinates of the points to be (sub-) sampled
        new_xyz: coordinates of the selected centroids to be sampled around
    
    OUTPUT:
        new_points: 
    '''
    # Generating indices for points within the query ball sphere of centroids
    idx = query_ball_point(radius, nsample, xyz, new_xyz) 
    # -> [num_blocks, ncentroids, nsample]
    # (Re-) indexing and selection of the grouped points according to ball query
    grouped_xyz = index_points(xyz, idx) 
    # -> [num_blocks, ncentroids, nsample, 3D coordinates]
    ''' NORMALIZATION OF SAMPLED AND GROUPED POINTS
    The sampled and grouped points are then normalized by subtracting the centroid
    coordinates to not include any bias towards certain positions. Therefore, the 
    resulting grouped_xyz_norm should have the sampled points around each centroid,
    with the respective centroid being the origin coordinates.
    '''
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    # -> [num_blocks, ncentroids, nsample, 3D coordinates]
    if points is not None:
        # Select sampled and grouped points from the input PC
        grouped_points = index_points(points, idx) 
        # Concatenate the normalized, sampled and grouped points with their coordinates
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        # -> [num_blocks, ncentroids, nsample, features+3D coordinates]
    else:
        new_points = grouped_xyz_norm
        
    '''
    RETURN
    new_xyz: 3D coordinates of the selected centroids 
        -> [num_blocks, num_centroids, 3D coord]
    new_points: Sampled and grouped points 
        -> [num_blocks, num_centroids, nsample, feat+3D coord]
    '''
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


#! ONLY NEEDED IF YOU DO SINGLE SCALE GROUPING
# Sampling and grouping with single scale instead of multiscale
def sample_and_group_all(xyz, points):
    """
    TODO: Come back and investigate why this is useful?
    
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
    ''' THEORY
    The PointNetSetAbstraction class encompasses all the operations for one set 
    abstraction, more specifically, one layer of the set asbtraction, which is performed
    multiple times. 
    This means, one forward pass of the PC through this function/ class does the following:
    1. Sampling and grouping points
        -> Sampling:    within a given radius a certain amount of points are selected around 
                        a specified amount of centroids
        -> Grouping:    after sampling K points around each centroid, the selected points
                        are grouped according to the centroids
    2. "PointNet"-Layer
        -> Within the point net layer the respective smampled and grouped points are passed
        through some MLP's to get abstractions of the respective points
        -> With each layer the amount of points is reduced, while increasing the radius 
        around the centroids and the amount fo features for each abstraction
    '''
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        ''' INPUT FOR INITIALIZATION
        - npoint: number of centroids to be sampled aorund from FPS
        - radius: radius around the centroids to sample within
        - nsample: number of points to be sampled around the centroids within the radius
        - in_channel: number of inputs for the first MLP of the set abstraction layer
        - mlp: list of in- and output channels of the serial MLP's
        - group_all: boolean determining whether points should be grouped locally
        '''
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            #? Why use conv2D here???
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        INPUT:
            xyz: input points position data [B, C, N]
                -> [num_blocks, 3D coordinates, num_points]
            points: input points data, [B, D, N]
                -> [num_blocks, num_features, num_points]
        OUTPUT:
            new_xyz: sampled points position data, [B, C, S]
                -> [num_blocks, 3D coordinates, num_sampled_points]
            new_points_concat: sample points feature data, [B, D', S]
                -> [num_blocks, num_features, num_sampled_points]
                #TODO: check if the number of sampled points is correct!
        """
        xyz = xyz.permute(0, 2, 1) #-> [num_blocks, num_points, 3D coordinates]
        if points is not None:
            points = points.permute(0, 2, 1) #-> [num_blocks, num_points, num_features]


        ''' SAMPLING AND GROUPING
        The sample_and_group function produces the sampled points with their features
        and dimensions concatenated, so they are ready for the first MLP pass.
        The returned variables have the following information and shape:
        new_xyz: 3D coordinates of selected centroids
            -> [num_blocks, num_centroids, 3d coord]
        new_points: sampled and grouped points
            -> [num_blocks, num_centroids, nsample, feat+3D coord]
        '''
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        '''
        new_xyz -> [num_blocks, num_points, 3D coordinates] / [B, npoint, C]
        new_points -> [num_blocks, num_points, num_sample_points, num_features] / [B, npoint, nsample, C+D]
        '''
        new_points = new_points.permute(0, 3, 2, 1) # -> [num_blocks, num_features, num_sample_points, num_points]
        
        
        ''' POINTNET LAYER
        In this for-loop the sampled and grouped points are passed through the "local"
        pointnet, to arrive at a more abstract representation of the input given. 
        Subsequently, as in line with the theory of the original PointNet, these
        representations are max-pooled for each local region, meaning we collapse
        the num_smaple_points dimension by choosing the maximum there. 
        '''
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        ''' RETURN
        new_xyz -> coordinates of the sampled and grouped points -> [num_blocks, 3D coords, num_points]
        new_points -> set-abstracted points -> [num_blocks, num_features, num_points]
        '''
        return new_xyz, new_points


# SET ABSTRACTION WITH MULTI-SCALE GROUPING: Sampling + Grouping + Pointnet Layer
class PointNetSetAbstractionMsg(nn.Module):
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
        # Module list of convolutional blocks with their respective layers
        self.conv_blocks = nn.ModuleList()
        # Module list of batch normalization layers after each convolutional layer
        self.bn_blocks = nn.ModuleList()
        # Looping trough number of MLP's
        for i in range(len(mlp_list)):
            # Creating Module lists for each convolutional layer
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
            #TODO: check here what dimensions the xyz and point features come out with!
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
        
        # 
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    '''
    Generally, the feature propagation in the PointNett++ is done via hierarchical 
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
            xyz1: point positions of less abstracted subsampled points
                -> [num_blocks, coords, num_points]
            xyz2: point positions of more abstracted subsampled points
                -> [num_blocks, coords, num_points]
            points1: less abstract representation of subsampled points
                -> [num_blocks, num_features, num_points]
            points2: more abstract representation of subsampled points
                -> [num_blocks, num_features, num_points]
            
        Output:
            new_points: 
                -> 
        """
        xyz1 = xyz1.permute(0, 2, 1) #[num_blocks, num_points, coords]
        xyz2 = xyz2.permute(0, 2, 1) #[num_blocks, num_points, coords]

        points2 = points2.permute(0, 2, 1) #[num_blocks, num_points, num_features]
        B, N, C = xyz1.shape # B: number of blocks | N: number of points less abstract
        _, S, _ = xyz2.shape # S: number of points more abstract

        # INTERPOLATION OF POINTS BETWEEN SET ABSTRACTION LAYERS
        if S == 1: #? Why this?
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # interpolation of points if feature levels are not equal
            dists = square_distance(xyz1, xyz2) #[8, 64, 16]
            dists, idx = dists.sort(dim=-1) #sort ascendingly according to closest centroids from higher abstraction layer
            # Only include the closest 3 centroids of higher abstraction representation
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [num_blocks, num_points_lower, 3]
            ''' CREATION OF DISTANCE-BASED WEIGHT FOR INTERPOLATION
            1. Inverse weight -> 1 over the distance
            2. Creating normalisation base by summing over all inverse distances
            3. Assigning dimensional weights according to individual 
            '''
            dist_recip = 1.0 / (dists + 1e-8) #1. | [num_blocks, num_points_lower, 3]
            #TODO: Maybe test and change the normalization
            #-> NOT to be normalized witht he sum of each distance, but rather normed by max of all selected distances?
            norm = torch.sum(dist_recip, dim=2, keepdim=True) #2. | [num_blocks, num_points_lower, 1]
            weight = dist_recip / norm #3. | [num_blocks, num_points_lower, 3]
            ''' DISTANCE-BASED FEATURE INTERPOLATION
            1. Pulling the actual features of the respective closest centroids for all points
            2. Multiplying with the determined inverse weight, to get feature estimations
            3. interpolated_points: points from lower SA layer with interpolated features from higher SA
                -> [num_blocks, num_points_lower, num_features_higher]
            '''
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) #[num_blocks, num_pointsSA, num_feat_SA]
        
        # I think this is the skip link concatenation
        if points1 is not None:
            points1 = points1.permute(0, 2, 1) #[num_blocks, num_points, num_features]
            new_points = torch.cat([points1, interpolated_points], dim=-1) #[num_blocks, num_points, num_features_sum]
        else:
            # for last feature propagation layer -> points1 = None
            new_points = interpolated_points

        # new_points -> concatenated points1 with interpolated points
        new_points = new_points.permute(0, 2, 1) #[num_blocks, num_features_sum, num_points]
        # Looping through MLP layers -> reducing feature size
        for i, conv in enumerate(self.mlp_convs):
            # Pull corresponding batch normalization layer
            bn = self.mlp_bns[i]
            # Convolve, Normalize, Activate
            new_points = F.relu(bn(conv(new_points)))
            
        # new_points: up-sampled point data [num_blocks, num_features, num_points]
        return new_points