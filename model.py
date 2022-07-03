from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, relu=True):
        super(MLP, self).__init__()
        self.linear = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.is_relu = relu
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        if self.is_relu:
            x = self.relu(x)
        return x


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        # layer 1: k -> 64
        self.mlp1 = MLP(k, 64)

        # layer 2:  64 -> 128
        self.mlp2 = MLP(64, 128)

        # layer 3: 128 -> 1024
        self.mlp3 = MLP(128, 1024)

        # fc 1024 -> 512
        self.fc1 = nn.Sequential(nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())

        # fc 512 -> 256
        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU())

        # fc 256 -> k*k (no batchnorm, no relu)
        self.fc3 = nn.Linear(256, k*k)
        # self.trainable_wts = nn.parameter.Parameter(torch.rand((256,k*k)))
        # self.trainable_bs = nn.parameter.Parameter(torch.rand((1,k*k)))

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)


    def forward(self, x):
        batch_size, _, num_points = x.shape
        # apply layer 1
        x = self.mlp1(x)

        # apply layer 2
        x = self.mlp2(x)

        # apply layer 3
        x = self.mlp3(x)

        # do maxpooling and flatten
        # x = self.maxpool(x).flatten()
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # apply fc layer 1
        x = self.fc1(x)

        # apply fc layer 2
        x = self.fc2(x)

        # apply fc layer 3
        x = self.fc3(x)

        #reshape output to a b*k*k tensor
        x = x.view(batch_size, self.k, self.k)

        # define an identity matrix to add to the output. This will help with the stability of the results since we want our transformations to be close to identity
        identity_mat = Variable(torch.eye(self.k, dtype=torch.float32)).repeat(batch_size,1,1) 

        x = x + identity_mat.cuda()
        x = x.view(-1, self.k, self.k)

        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform='True'):
        super(PointNetfeat, self).__init__()
        self.global_feat = global_feat

        # Use TNet to apply transformation on input and multiply the input points with the transformation
        self.TNet_input = TNet(k=3)

        # layer 1:3 -> 64
        self.conv1 = MLP(3, 64)
    
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        self.feature_transform=feature_transform

        if self.feature_transform == "True":
            self.TNet_feature = TNet(k=64)

        # layer2: 64 -> 128
        self.conv2 = MLP(64, 128)

        # layer 3: 128 -> 1024 (no relu)
        self.conv3 = MLP(128, 1024, relu=False)

        # ReLU activation
        self.relu = nn.ReLU()



    def forward(self, x):
        batch_size, _, num_points = x.shape
        # input transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        trans_input = self.TNet_input(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_input)
        x = x.transpose(2,1)
        # apply layer 1
        x = self.conv1(x)

        # feature transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        if self.feature_transform == "True":
            trans_feat = self.TNet_feature(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None
        
        local_feat = x

        # apply layer 2
        x = self.conv2(x)

        # apply layer 3
        x = self.conv3(x)

        # apply maxpooling
        x = torch.max(x, 2, keepdim=True)[0]

        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat == True: # For classification 
            x = x.squeeze(2)
            return x, trans_input, trans_feat
        else: # For segmentation
            x = torch.concat([x.repeat(1,1,num_points), local_feat], axis=1)
            return x, trans_input, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=True):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        print("************* FEATURE TRANSFORM", feature_transform, " *************")
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x, trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.num_classes = num_classes
        print("************* FEATURE TRANSFORM", feature_transform, " *************")
        # get global features + point features from PointNetfeat
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)

        # layer 1: 1088 -> 512
        self.conv1 = MLP(1088, 512)

        # layer 2: 512 -> 256
        self.conv2 = MLP(512, 256)

        # layer 3: 256 -> 128
        self.conv3 = MLP(256, 128)

        # layer 4:  128 -> k (no ru and batch norm)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        # ReLU activation
        self.relu = nn.ReLU()

    
    def forward(self, x):
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        batch_size, _, num_points = x.shape
        x_local, trans_input, trans_feat = self.feat(x) 

        x = self.conv1(x_local)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = F.log_softmax(x, dim=1)
        # x = x.view(batch_size, num_points, self.num_classes)
        return x, trans_input, trans_feat



def feature_transform_regularizer(trans):
    batch_size, feature_size, _ = trans.shape
    
    # compute I - AA^t
    identity_mat = Variable(torch.eye(feature_size, dtype=torch.float32)).cuda()
    reg_term = identity_mat - torch.bmm(trans, trans.transpose(2,1))
    
    # compute norm
    reg_term = torch.linalg.matrix_norm(reg_term, ord='fro', dim=(1,2))
    # compute mean norms and return
    return reg_term.mean()



if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500)).cuda()
    trans = TNet(k=3).cuda()
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500)).cuda()
    trans = TNet(k=64).cuda()
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True).cuda()
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False).cuda()
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes = 5).cuda()
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes = 3).cuda()
    out, _, _ = seg(sim_data)
    print('seg', out.size())

