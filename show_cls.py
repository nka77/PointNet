from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--feature_transform', default='True', help="use feature transform")
parser.add_argument('--dataset', default='/scratch/nka77/shapenet/', help="dataset path")

opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    split='test',
    npoints=2500)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

print("total pt clouds:", len(test_dataset))
class_keys = list(test_dataset.classes.keys())
classifier = PointNetCls(num_classes=len(test_dataset.classes), feature_transform=opt.feature_transform)
classifier.cuda()

checkpoint = torch.load(opt.model)
classifier.load_state_dict(checkpoint['model'])
classifier.eval()

def get_critical_point(input_, target, model, batch):
    model = model.feat
    trans_input = model.TNet_input(input_)
    x = input_.transpose(2,1)
    x = torch.bmm(x, trans_input)
    x = x.transpose(2,1)
    x = model.conv1(x)

    if opt.feature_transform == "True":
        trans_feat = model.TNet_feature(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2,1)
    else:
        trans_feat = None
    
    local_feat = x
    x = model.conv2(x)
    x = model.conv3(x)

    # for each feat out of 1024, get indices where value is maximum:
    critical_points = torch.max(x, 2, keepdim=True)[1].squeeze(2)

    # pick first pt object out of the batch:
    pts = input_[0,:,:].transpose(1,0).cpu().numpy()
    critical_points = critical_points[0,:].cpu().numpy()
    class_ = target.cpu()[0]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.title(str(class_keys[class_]))

    ax2 = fig.add_subplot(1, 2, 1, projection = '3d')
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]
    color = np.vstack([xs,ys,zs]).transpose()
    color += 1
    color /= 2
    color = np.clip(color, 0, 1)
    ax2.scatter(xs, ys, zs, c=color)
    ax2.set_axis_off()
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)


    ax1 = fig.add_subplot(1, 2, 2, projection = '3d')
    xs = pts[critical_points, 0]
    ys = pts[critical_points, 1]
    zs = pts[critical_points, 2]
    color = np.vstack([xs,ys,zs]).transpose()
    color += 1
    color /= 2
    color = np.clip(color, 0, 1)
    ax1.scatter(xs, ys, zs, c=color)
    ax1.set_axis_off()
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    
    plt.savefig("cls/"+str(batch)+".png")
    plt.close('all')
    return


## TEST set accuracy
total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim= 1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])
        a = 0

        get_critical_point(points, target,  classifier, i)     


    accuracy = 100 * (total_targets == total_preds).sum() / len(test_dataset)
    print('************* TEST ACCURACY = {:.2f}% *************'.format(accuracy))
    confusion_mat = confusion_matrix(total_targets, total_preds)
    for i in range(len(test_dataset.classes)):
        tp = confusion_mat[i,i]
        fp = confusion_mat[:i,i].sum() + confusion_mat[i+1:,i].sum()
        fn = confusion_mat[i,:i].sum() + confusion_mat[i,i+1:].sum()
        f1_score = 2 *tp /(2*tp + fp + fn)
        print('{}: {:.2f}'.format(class_keys[i], f1_score), end='\t')
    print('\n')
    



## TRAIN set accuracy
dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    npoints=2500)

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(train_dataloader, 0):
        #TODO
        # calculate average classification accuracy
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim= 1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])
        a = 0
    accuracy = 100 * (total_targets == total_preds).sum() / len(dataset)
    print('************* TRAIN ACCURACY = {:.2f}% *************'.format(accuracy))
    confusion_mat = confusion_matrix(total_targets, total_preds)
    for i in range(len(test_dataset.classes)):
        tp = confusion_mat[i,i]
        fp = confusion_mat[:i,i].sum() + confusion_mat[i+1:,i].sum()
        fn = confusion_mat[i,:i].sum() + confusion_mat[i,i+1:].sum()
        f1_score = 2 *tp /(2*tp + fp + fn)
        print('{}: {:.2f}'.format(class_keys[i], f1_score), end='\t')
    print('\n')


