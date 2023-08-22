import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, ncentroids, radius, samples_around_centroid, sa_mlps, fp_mlps, dropout, dropout_prob):
        super(get_model, self).__init__()
        # SET ABSTRACTION
        self.sa1 = PointNetSetAbstraction(ncentroids[0], 
                                          radius[0], 
                                          samples_around_centroid, 
                                          9 + 3,
                                          sa_mlps[0], 
                                          False)
        self.sa2 = PointNetSetAbstraction(ncentroids[1], 
                                          radius[1], 
                                          samples_around_centroid,
                                          sa_mlps[0][-1] + 3, #64 + 3
                                          sa_mlps[1], 
                                          False)
        self.sa3 = PointNetSetAbstraction(ncentroids[2], 
                                          radius[2], 
                                          samples_around_centroid, 
                                          sa_mlps[1][-1] + 3, #128 + 3
                                          sa_mlps[2], 
                                          False)
        self.sa4 = PointNetSetAbstraction(ncentroids[3], 
                                          radius[3], 
                                          samples_around_centroid, 
                                          sa_mlps[2][-1] + 3, #256 + 3
                                          sa_mlps[3], 
                                          False)
        self.sa5 = PointNetSetAbstraction(ncentroids[4], 
                                          radius[4], 
                                          samples_around_centroid, 
                                          sa_mlps[3][-1] + 3, #512 + 3
                                          sa_mlps[4], 
                                          False)
        self.sa6 = PointNetSetAbstraction(ncentroids[5], 
                                          radius[5], 
                                          samples_around_centroid, 
                                          sa_mlps[4][-1] + 3, #1024 + 3
                                          sa_mlps[5], 
                                          False)
        """ self.sa7 = PointNetSetAbstraction(ncentroids[6], 
                                          radius[6], 
                                          samples_around_centroid, 
                                          sa_mlps[5][-1] + 3, #1024 + 3
                                          sa_mlps[6], #2048
                                          False)
         """
        # FEATURE PROPAGATION
        """ self.fp7 = PointNetFeaturePropagation(sa_mlps[-1][-1] + sa_mlps[-2][-1], #2048 + 1024
                                              fp_mlps[0]) """
        self.fp6 = PointNetFeaturePropagation(sa_mlps[-2][-1] + sa_mlps[-3][-1], #1024 + 1024
                                              fp_mlps[1]) 
        self.fp5 = PointNetFeaturePropagation(sa_mlps[-3][-1] + sa_mlps[-4][-1], #1024 + 512
                                              fp_mlps[2]) 
        self.fp4 = PointNetFeaturePropagation(sa_mlps[-4][-1] + sa_mlps[-5][-1], #512 + 256
                                              fp_mlps[3]) 
        self.fp3 = PointNetFeaturePropagation(sa_mlps[-5][-1] + sa_mlps[-6][-1], #256 + 128
                                              fp_mlps[4]) 
        self.fp2 = PointNetFeaturePropagation(sa_mlps[-5][-1] + sa_mlps[-7][-1], #256 + 64
                                              fp_mlps[5])
        self.fp1 = PointNetFeaturePropagation(sa_mlps[-6][-1], 
                                              fp_mlps[6])
        # CLASSIFICATION
        self.conv1 = nn.Conv1d(sa_mlps[-6][-1], sa_mlps[-6][-1], 1)
        self.bn1 = nn.BatchNorm1d(sa_mlps[-6][-1])
        if dropout:
            self.drop1 = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv1d(sa_mlps[-6][-1], num_classes, 1)


    def forward(self, xyz, loss_function, dropout):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        # SET ABSTRACTION
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)
        # FEATURE PROPAGATION
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # CLASSIFICATION
        if dropout:
            x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        else:
            x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # Specifying model output according to chosen loss function
        if loss_function == "CE-Loss" or loss_function == "BCE-Loss":
            y = x #logits
            probs = F.softmax(x, dim=1) #probabilities
        elif loss_function == "NLL-Loss":
            y = F.log_softmax(x, dim=1) #log-probabilities
            probs = F.softmax(x, dim=1) #probabilities
        else:
            print("Loss function not specified correctly, unsure what output to deliver...")
        # Permuting for correct shape
        y = y.permute(0, 2, 1)
        probs = probs.permute(0, 2, 1)
        
        #! Changed from l4_points to l5_points to give back highest abstract represntation
        return y, l5_points, probs


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self,loss_function, pred, target, trans_feat, weight):
        # Cross Entropy Loss
        if loss_function == 'CE-Loss':
            total_loss = F.cross_entropy(pred, target, weight=weight)
        # Negative-Log-Likelihood-Loss
        elif loss_function == 'NLL-Loss':
            total_loss = F.nll_loss(pred, target, weight=weight)
        # Binary Cross Entropy Loss with Logits and Weights
        elif loss_function == 'BCE-Loss':
            total_loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        else:
            print("No loss function has been specified in the train_config.yaml...")
        

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))