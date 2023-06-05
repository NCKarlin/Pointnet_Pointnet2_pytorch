import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        # INITIALIZING SET ABSTRACTION LEVELS WITH MULTI-SCALE GROUPING
        '''
        Some interesting points concerning the set abstraction levels:
        - npoints: the number of centroids points to be sampled decreases per abstraction
            -> less centroids, less locality, less details -> more global
        - radius_list: the radii in the lsit increase per abstraction layer
            -> because of less centroids, you need bigger radii to get more of the hood
        - n_sample_list: the number of sample poitns per ball query stay the same
            -> not incorporating any bias for any abstraction level
        - in_channel: increases for every abstraction layer 
            -> more and more abstract feature representation of the input pc
        - mlp_list: number of MLP's stay the same, but dimensions increase
        '''
        self.sa1 = PointNetSetAbstractionMsg(1024, #npoints: number of sampled centroids
                                             [0.05, 0.1], #radius_list: radii for ball query
                                             [16, 32], #nsample_list: max samples for each ball query
                                             9, #in_channel: number of features for SA layer
                                             [[16, 16, 32], [32, 32, 64]]) #mlp_list: MLP's with in & out layer dim's
        self.sa2 = PointNetSetAbstractionMsg(256, #npoints
                                             [0.1, 0.2], #radius_list
                                             [16, 32], #nsample_list
                                             32+64, #in_channel
                                             [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, #npoints
                                             [0.2, 0.4], #radius_list
                                             [16, 32], #nsample_list
                                             128+128, #in_channel
                                             [[128, 196, 256], [128, 196, 256]]) #mlp_list
        self.sa4 = PointNetSetAbstractionMsg(16, #npoints
                                             [0.4, 0.8], #radius_list
                                             [16, 32], #nsample_list
                                             256+256, #in_channel
                                             [[256, 256, 512], [256, 384, 512]]) #mlp_list
        
        # INITIALIZING FEATURE PROPAGATION FOR SET SEGMENTATION
        '''
        For the feature propagation the main interesting points are:
        - 
        '''
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, #in_channel: number of input channels  
                                              [256, 256]) #mlp: MLP's with in & out layer dim's
        self.fp3 = PointNetFeaturePropagation(128+128+256, #in_channel
                                              [256, 256]) #mlp
        self.fp2 = PointNetFeaturePropagation(32+64+256, #in_channel
                                              [256, 128]) #mlp
        self.fp1 = PointNetFeaturePropagation(128, #in_channel
                                              [128, 128, 128]) #mlp
        
        # LAST UNIT POINTNET FOR PER-POINT SCORES
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # Random input dropout
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    # INPUT: xyz - point cloud to be classified
    def forward(self, xyz):
        # Assigning coordinates and entire pc to relevant variables 
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        # FORWARD PASSING THROUGH THE SET ABSTRACTION LAYERS
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #l1_xyz: 1024x3 / l1_points: 1024x64
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #l2_xyz: 256x3 / l2_points: 256x128
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) #l3_xyz: 64x3 / l3_points: 64x256
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) #l4_xyz: 16x3 / l4_points: 16x512

        # FORWARD PASSING THROUGH FEATURE PROPAGATION LAYERS 
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) #l3_points: 64x256
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) #l2_points: 256x256
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) #l1_points: 1024x128
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) #l0_points: Nx128

        # LAST UNIT POINTNET FOR SEMANTIC SEGMENTATION
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x) #x: Nxnum_classes 
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        # Returning the semantic segmentation & most abstract representation?
        return x, l4_points


# NEGATIVE LOG LIKELIHOOD LOSS
# Loss function mostly used for multi-class classification
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))