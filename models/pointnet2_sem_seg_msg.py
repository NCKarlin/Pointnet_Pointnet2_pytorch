import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, 
                 num_classes,
                 ncentroids,
                 radius,
                 samples_around_centroid,
                 sa_mlps,
                 fp_mlps,
                 dropout,
                 dropout_prob):
        super(get_model, self).__init__()
        # SET ABSTRACTION LAYERS
        self.sa1 = PointNetSetAbstractionMsg(ncentroids[0], #2048
                                             radius[0], #[0.2, 0.4]
                                             samples_around_centroid, #[16, 32]
                                             9, #in_channel: RGBXYZ XYZ
                                             sa_mlps[0]) #[[16, 32, 64], [32, 32, 64]]
        self.sa2 = PointNetSetAbstractionMsg(ncentroids[1], #512
                                             radius[1], #[0.4, 0.6]
                                             samples_around_centroid, #[16, 32]
                                             sa_mlps[0][0][-1] + sa_mlps[0][1][-1], #64 + 64
                                             sa_mlps[1]) #[[64, 64, 128], [64, 96, 128]]
        self.sa3 = PointNetSetAbstractionMsg(ncentroids[2], #256
                                             radius[2], #[0.8, 1.0]
                                             samples_around_centroid, #[16, 32]
                                             sa_mlps[1][0][-1] + sa_mlps[1][1][-1], #128 + 128
                                             sa_mlps[2]) #[[128, 196, 256], [128, 196, 256]]
        self.sa4 = PointNetSetAbstractionMsg(ncentroids[3], #128
                                             radius[3], #[1.2, 1.4]
                                             samples_around_centroid, #[16, 32]
                                             sa_mlps[2][0][-1] + sa_mlps[2][1][-1], #256 + 256
                                             sa_mlps[3]) #[[256, 256, 512], [256, 384, 512]]
        self.sa5 = PointNetSetAbstractionMsg(ncentroids[4], #64
                                            radius[4], #[1.6, 1.8]
                                            samples_around_centroid, #[16, 32]
                                            sa_mlps[3][0][-1] + sa_mlps[3][1][-1], #512 + 512
                                            sa_mlps[4]) #[[512, 512, 1024], [512, 512, 1024]]
        
        # INITIALIZING FEATURE PROPAGATION FOR SET SEGMENTATION
        #TODO: Adjust the in channel to all the FP MLP's so model can run
        #512+512+256+256, #in_channel: number of input channels  
        self.fp5 = PointNetFeaturePropagation(sa_mlps[-1][0][-1] + sa_mlps[-1][1][-1] + sa_mlps[-2][0][-1] + sa_mlps[-2][1][-1], #1024+1024+512+512 [3072]
                                              fp_mlps[0]) #[2048, 1024]
        self.fp4 = PointNetFeaturePropagation(sa_mlps[-2][0][-1] + sa_mlps[-2][1][-1] + sa_mlps[-3][0][-1] + sa_mlps[-3][1][-1], #512+512+256+256 [1536]
                                              fp_mlps[1]) #[512, 512]
        self.fp3 = PointNetFeaturePropagation(sa_mlps[-3][0][-1] + sa_mlps[-3][1][-1] + sa_mlps[-4][0][-1] + sa_mlps[-4][1][-1], #256+256+128+128 [768]
                                              fp_mlps[2]) #[256, 256]        
        self.fp2 = PointNetFeaturePropagation(sa_mlps[-4][0][-1] + sa_mlps[-4][1][-1] + sa_mlps[-5][0][-1] + sa_mlps[-5][1][-1], #128+128+64+64 [384]
                                              fp_mlps[3]) #[256, 128]
        self.fp1 = PointNetFeaturePropagation(fp_mlps[-1][-1], #128
                                              fp_mlps[4]) #[128, 128, 128]
        
        # LAST UNIT POINTNET FOR PER-POINT SCORES
        self.conv1 = nn.Conv1d(fp_mlps[-1][-1], fp_mlps[-1][-1], 1)
        self.bn1 = nn.BatchNorm1d(fp_mlps[-1][-1])
        if dropout:
            self.drop1 = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv1d(fp_mlps[-1][-1], num_classes, 1)
    
    # INPUT: xyz - point cloud to be classified
    def forward(self, xyz, loss_function, dropout):
        # ASSIGNMENT OF POSITIONAL AND FEATURE DATA
        l0_points = xyz #batch x 9 x npoint
        l0_xyz = xyz[:,:3,:] #batch x 3 x npoints

        # FORWARD PASSING THROUGH THE SET ABSTRACTION LAYERS
        # lX_xyz: num_blocks x dim x num_centroids | lX_points: num_blocks x num_features x num_centroids
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #l1_xyz: 4x3x2048 | l1_points: 4x128x2048
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #l2_xyz: 4x3x512  | l2_points: 4x256x512
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) #l3_xyz: 4x3x256  | l3_points: 4x512x256
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) #l4_xyz: 4x3x128  | l4_points: 4x1024x128
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) #l5_xyz: 4x3x64   | l5_points: 4x2048x64

        # FORWARD PASSING THROUGH FEATURE PROPAGATION LAYERS 
        # lX_points: num_blocks x num_features x num_centroids
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  #l4_points: 4x1024x128
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  #l3_points: 4x512x256
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  #l2_points: 4x256x512
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  #l1_points: 8x128x1024
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       #l0_points: 8x128x1024

        # LAST UNIT POINTNET FOR SEMANTIC SEGMENTATION
        if dropout:
            x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        else:
            x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # Specifying model ouotput according to loss function
        if loss_function == "CE-Loss" or loss_function == "BCE-Loss":
            y = x
            sigmoid = nn.Sigmoid()
            probs = sigmoid(x) #make probs of logits
        elif loss_function == "NLL-Loss":
            y = F.log_softmax(x, dim=1)
            probs = F.softmax(x, dim=1)
        else:
            print("Loss-function not specified correctly, unsure what output to deliver...")
        # Permuting for correct shape
        y = y.permute(0, 2, 1)
        probs = probs.permute(0, 2, 1)
        
        # Returning the semantic segmentation & most abstract representation?
        return x, l4_points, probs


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, loss_function, pred, target, trans_feat, weight):
        # Cross Entropy Loss
        if loss_function == 'CE-Loss':
            total_loss = F.cross_entropy(pred, target, weight=weight)
        # Binary Cross Entropy Loss with logits (instead of porbabilities)
        elif loss_function == 'BCE-Loss':
            total_loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        # Negative-Log-Likelihood Loss
        elif loss_function == 'NLL-Loss':
            total_loss = F.nll_loss(pred, target, weight=weight)
        else:
            print("Loss function specification in train_config.yaml improper...")

        return total_loss


if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))