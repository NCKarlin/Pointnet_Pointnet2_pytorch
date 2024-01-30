import torch.nn as nn
import torch.nn.functional as F
from models.fragrec_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, 
                 num_classes,
                 ncentroids_min,
                 radius_min,
                 samples_around_centroid,
                 sa_mlps_min,
                 fp_mlps_min, 
                 loss_function):
        super(get_model, self).__init__()
        
        # SET ABSTRACTION LAYERS
        self.sa1 = PointNetSetAbstractionMsg(ncentroids_min[0],
                                             radius_min[0],
                                             samples_around_centroid,
                                             9, #in_channel: RGBXYZ XYZ
                                             sa_mlps_min[0])
        self.sa2 = PointNetSetAbstractionMsg(ncentroids_min[1],
                                             radius_min[1],
                                             samples_around_centroid,
                                             sa_mlps_min[0][0][-1] + sa_mlps_min[0][1][-1],
                                             sa_mlps_min[1])
        
        # FEATURE PROPAGATION LAYERS
        self.fp2 = PointNetFeaturePropagation(sa_mlps_min[-1][0][-1]+sa_mlps_min[-1][1][-1]+sa_mlps_min[-2][0][-1]+sa_mlps_min[-2][1][-1],
                                              fp_mlps_min[0])
        self.fp1 = PointNetFeaturePropagation(fp_mlps_min[-1][0],
                                              fp_mlps_min[1])
        
        # LAST UNIT POINTNET FOR PER-POINT SCORES (output prep depenent on loss function)
        self.final_conv1 = nn.Conv1d(fp_mlps_min[-1][-1], fp_mlps_min[-1][-1], 1)
        self.final_bn = nn.BatchNorm1d(fp_mlps_min[-1][-1])
        if loss_function == 'BCE-Loss':
            self.final_conv2 = nn.Conv1d(fp_mlps_min[-1][-1], 1, 1)
        elif loss_function == 'CE-Loss':
            self.final_conv2 = nn.Conv1d(fp_mlps_min[-1][-1], num_classes, 1)
    
    # INPUT: point_data - point cloud to be classified
    def forward(self, points_data, loss_function):
        # ASSIGNMENT OF POSITIONAL AND FEATURE DATA
        l0_points = points_data #batch x 9 x npoint
        l0_xyz = points_data[:,:3,:] #batch x 3 x npoints

        # FORWARD PASSING THROUGH THE SET ABSTRACTION LAYERS
        # lX_xyz: num_blocks x 3 x num_centroids | lX_points: num_blocks x num_features x num_centroids
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # FORWARD PASSING THROUGH FEATURE PROPAGATION LAYERS 
        # lX_points: num_blocks x num_features x num_centroids
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # LAST UNIT POINTNET FOR PER POINT SCORES
        norm_out = F.relu(self.final_bn(self.final_conv1(l0_points)))
        logits = self.final_conv2(norm_out)
        
        # LOSS FUNCTION PREPARATION
        if loss_function == 'BCE-Loss': #takes raw model logit output as input
            model_output = logits
            sigmoid = nn.Sigmoid()
            model_output_probs = sigmoid(logits)
        #TODO: Prepare network output properly for CE-Loss and also adjust structure of final unit point net!!!
        elif loss_function == 'CE-Loss': #takes log-probabilities as input
            model_output = logits
            model_output_probs = F.softmax(logits, dim=1)
        
        # PERMUTATION FOR CORRECT SHAPE
        model_output = model_output.permute(0, 2, 1)
        model_output_probs = model_output_probs.permute(0, 2, 1)
            
        # TODO: adjust return accordingly + do we need l2_points here as return?
        # Returning the semantic segmentation, most abstract representation, model output probabilities
        return model_output, model_output_probs


class get_loss(nn.Module):
    
    def __init__(self):
        super(get_loss, self).__init__()
    
    def forward(self, loss_function, pred, target, weight):
        # Cross Entropy Loss - input is softmaxed model output logits 
        if loss_function == 'CE-Loss':
            total_loss = F.cross_entropy(pred, target, weight=weight)
        # Binary Cross Entropy Loss with logits (instead of probabilities)
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