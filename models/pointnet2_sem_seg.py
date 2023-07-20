import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        
        ''' SET ABSTRACTION
        For the initiation of the PointNetSetAbstraction layers the following inputs:
        - npoint: number of centroids to sample around 
        - radius: local region radius to sample within
        - nsample: max sample number for the local region
        - in_channel: # of channels for the first concatenated mlp input
        - mlp: list of input and output dimensions of the MLP's for the SA layer
        - group_all: boolean determining which sampling and grouping function to use
        '''
        self.sa1 = PointNetSetAbstraction(1024, #npoint
                                          0.1, #radius
                                          32, #nsample
                                          9 + 3, #in_channel (feat + coord)
                                          [32, 32, 64], #mlp
                                          False) #group_all
        self.sa2 = PointNetSetAbstraction(256, #npoint
                                          0.2, #radius
                                          32, #nsample
                                          64 + 3, #in_channel (feat + coord)
                                          [64, 64, 128], #mlp
                                          False) #group_all
        self.sa3 = PointNetSetAbstraction(64, #npoint
                                          0.4, #radius
                                          32, #nsample
                                          128 + 3, #in_channel (feat + coord)
                                          [128, 128, 256], #mlp
                                          False) #group_all
        self.sa4 = PointNetSetAbstraction(16, #npoint
                                          0.8, #radius
                                          32, #nsample
                                          256 + 3, #in_channel (feat + coord)
                                          [256, 256, 512], #mlp
                                          False) #group_all
        
        ''' FEATURE PROPAGATION
        This is where the skip link concatentation takes place, which concatenates the 
        features to be propagated wiht the features from the abstraction layer above.
        For the initiation of the PointNetFeaturePropagation layer are:
        - in_channel: number of input channels for the first MLP layer
        - mlp: number of in- and output-channels for the MLPs in the FP layer
        '''
        self.fp4 = PointNetFeaturePropagation(768, #in_channel -> 512 + 256
                                              [256, 256]) #mlp
        self.fp3 = PointNetFeaturePropagation(384, #in_channel -> 256 + 128
                                              [256, 256]) #mlp
        self.fp2 = PointNetFeaturePropagation(320, #in_channel -> 256 + 64 
                                              [256, 128]) #mlp 
        self.fp1 = PointNetFeaturePropagation(128, #in_channel -> 128
                                              [128, 128, 128]) #mlp
        
        ''' LAST UNIT POINTNET FOR CLASSIFICATION
        For the last unit pointnet that is used for the final classification of the
        points, there is one convolution to be done, with its subsequent batch 
        normalization, before a dropout layer is inserted to prevent the model from over-
        fitting. The last convolutional layer is then used for the actual classification
        into the respective classes.
        '''
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5) #possibly another HP by varying the dropout ratio
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        '''
        The shapes for the following lines are:
        - xyz (input point cloud to be classified) -> [8, 9, 4096] (blocks (?), channels, num_points)
        - l0_points (entire pc which enters the first layer) -> [8, 9, 4096]
        - l0_xyz (coordinates of the points) -> [8, 3, 4096]
        '''
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        ''' SET ABSTRACTION
        The general shape of the set abstraction layers is: 
        [num_blocks, num_features, num_points]
        -> number of points decrease while features increase with abstraction
        The specific shapes for the set abstraction layers are:
        - l1_xyz (coordinates of first set abstraction) -> [8, 3, 1024]
        - l1_points (features of first set abstraction) -> [8, 64, 1024]
        - l2_xyz (coordinates of second set abstraction) -> [8, 3, 256]
        - l2_points (features of second set abstraction) -> [8, 128, 256]
        - l3_xyz (coordinates of third set abstraction) -> [8, 3, 64]
        - l3_points (features of third set abstraction) -> [8, 256, 64]
        - l4_xyz (coordinates of fourth set abstraction) -> [8, 3, 16]
        - l4_points (features of fourth set abstraction) -> [8, 512, 16]
        '''
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        ''' FEATURE PROPAGATION
        The general shape for the feature propagation layers is:
        [num_blocks, num_features, num_points]
        -> number of features decrease, while the number of points increases
        The shapes for the feature propagation layers are:
        - l3_points (first feature propagation output) -> [8, 256, 64]
        - l2_points (second feature propagation output) -> [8, 256, 256]
        - l1_points (third feature propagation output) -> [8, 128, 1024]
        - l0_points (fourth feature propagation output) -> [8, 128, 4096]
        '''
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        ''' LAST UNIT POINTNET 
        The droput layer is inserted to better regularize the NN. This basically means, 
        that before handing the values to the last classification layer, some of the 
        values/ values from neurons are set to zero, so that the classifier does not
        rely on only one or a few features for its classification, but rather optimizes
        its strategy towards using all the features, as during training some of them 
        constantly randomly disappear. 
        Therefore, the implementation of random input dropout leads to less overfitting
        of the model, and a better distribution of the prediction ability among all
        features.
        The main downside when implementing dropout is the dilution of the cost 
        function (models error over entire dataset). This means, that the repro-
        duction of the cost function becomes more difficult if randomly some 
        neurons are dropped out all the time. Hence, one suggestion is to run
        the model first without dropout to make sure the cost function is 
        monotonically decreasing and then implement dropout and hope for better
        performance.
        
        TODO: identify and define these two variables/ layers
        The shape for the dropout layer are:
        - x () -> [8, 128, 4096]
        - x () -> [8, 2, 4096]
        '''
        # The shapes for the dropout layer are:
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)))) # [8, 128, 4096]
        x = self.conv2(x) # [8, 2, 4096]

        ''' PER POINT PREDICTIONS
        The log-softmax function is simply the logarithm of the sosftmax function. 
        This means, that the probabilities produced by the softmax function, can then
        be represented on a logarithmic scale. This has the advantage, that you can then
        add the log probabilities of independent events, insetad of multiplying these
        and producing ever smaller numbers, which is practical for computations.
        Therefore, it can provide a better numerical performance and also help with the
        gradient optimization, as the numbers do not explode in any direction.
        
        The shapes for the log-softmax layer are:
        '''
        y = F.log_softmax(x, dim=1) #torch.Size([8, 2, 4096])
        y = y.permute(0, 2, 1) #torch.Size([8, 4096, 2])

        ''' POINT PROBABILITIES FOR AUC-SCORE
        The softmax function converts the output values (predictions) of the model  to 
        the respective probabilities of the respective classes. It basically represents
        a smooth version of the winner-tkaes-it-all principle, by using the values as
        exponents and dividing by the sum to normalize them. 
        In our case the probabilities are then used for ROC-AUC-score for the evaluation.
        
        The shapes for the softmax layer (for probabilities) are:   
        '''
        probs = F.softmax(x, dim=1) #torch.Size([8, 2, 4096])
        probs = probs.permute(0, 2, 1) #torch.Size([8, 4096, 2])

        ''' RETURN
        The shape of the returns are the:
        - y (class predictions for input pc) -> [8, 4096, 2]
        - l4_points (most abstract representation of input PC) -> [8, 512, 16]
        - probs (probabilities of class predictions for input PC) -> [8, 4096, 2]
        '''
        return y, l4_points, probs


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        '''
        The input [shapes] for the loss function:
        - pred (point predictions) -> [32.768, 2]
        - target (point classes) -> [32.768]
        - trans_feat (features of highest set abstraction) -> [8, 512, 16]
        - weight (class weights) -> [2]
        
        The output [shape] of the loss function:
        total_loss (total loss for the respective batch) -> [1]
        '''
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))