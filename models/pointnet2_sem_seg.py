import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        # SET ABSTRACTION
        self.sa1 = PointNetSetAbstraction(2048, 
                                          0.1, 
                                          64, 
                                          9 + 3,
                                          [32, 64, 128], 
                                          False)
        self.sa2 = PointNetSetAbstraction(512, 
                                          0.2, 
                                          64, 
                                          128 + 3, 
                                          [128, 128, 256], 
                                          False)
        self.sa3 = PointNetSetAbstraction(128, 
                                          0.4, 
                                          64, 
                                          256 + 3, 
                                          [256, 256, 512], 
                                          False)
        self.sa4 = PointNetSetAbstraction(32, 
                                          0.8, 
                                          64, 
                                          512 + 3, 
                                          [512, 512, 1024], 
                                          False)
        # FEATURE PROPAGATION
        self.fp4 = PointNetFeaturePropagation(1536, #512+256 | 1024+512
                                              [512, 512]) 
        self.fp3 = PointNetFeaturePropagation(768, #256+128 | 512+256
                                              [512, 256]) 
        self.fp2 = PointNetFeaturePropagation(384, #256+64 | 256+128
                                              [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, 
                                              [128, 128, 128])
        # CLASSIFICATION
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)


    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        # SET ABSTRACTION
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # FEATURE PROPAGATION
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # CLASSIFICATION
        #x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = F.relu(self.bn1(self.conv1(l0_points))) #NO DROPOUT
        x = self.conv2(x)
        y = F.log_softmax(x, dim=1) #torch.Size([8, 2, 4096])
        y = y.permute(0, 2, 1) #torch.Size([8, 4096, 2])
        # TODO: Change this to Sigmoid for binary classsification instead of softmax
        #probs = F.sigmoid(x, dim=1) #torch.Size([8, 2, 4096])
        probs = probs.permute(0, 2, 1) #torch.Size([8, 4096, 2])

        return y, l4_points, probs


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        # Negative-Log-Likelihood loss
        #total_loss = F.nll_loss(pred, target, weight=weight)
        # Cross Entropy Loss
        #total_loss = F.cross_entropy(pred, target, weight=weight)
        # Binary Cross Entropy Loss with weights
        total_loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))