from __future__ import print_function
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', default='True', help="use feature transform")
parser.add_argument('--save_dir', default='pretrained_networks', help='directory to save model weights')
parser.add_argument('--lr', default=0.001, help='learning rate')

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    npoints=opt.num_points)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

val_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    split='val',
    npoints=opt.num_points)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(num_classes=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=float(opt.lr), betas=(0.9, 0.999))
classifier.cuda()

num_batch = len(dataloader)

for epoch in range(opt.nepoch):
    classifier.train()
    for i, data in enumerate(tqdm(dataloader, desc='Batches', leave = False), 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        #TODO
        optimizer.zero_grad()
        # perform forward and backward paths, optimize network
        preds, trans_input, trans_feat = classifier(points)
        # print(preds.shape, trans_input.shape, trans_feat.shape)
        loss = nn.CrossEntropyLoss()(preds, target)
        loss += feature_transform_regularizer(trans_input) * 0.001
        if opt.feature_transform == "True":
            loss += feature_transform_regularizer(trans_feat) * 0.001
        
        loss.backward()
        optimizer.step()

    model_file = 'latest_classification_v1.pt'
    if opt.feature_transform == "False":
        model_file = 'latest_classification_notrans_v1.pt'
    
    torch.save({'model':classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, os.path.join('pretrained_networks', model_file))

    classifier.eval()
    total_preds = []
    total_targets = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            preds, _, _ = classifier(points)
            pred_labels = torch.max(preds, dim= 1)[1]

            total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
            total_targets = np.concatenate([total_targets, target.cpu().numpy()])
            a = 0
        accuracy = 100 * (total_targets == total_preds).sum() / len(val_dataset)
        print('Epoch:{} Accuracy = {:.2f}%'.format(epoch, accuracy))





