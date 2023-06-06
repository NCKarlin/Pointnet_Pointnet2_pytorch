import argparse
import os
from data_utils.S3DISDataLoader import FracDataset, S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

def worker_init(x):
    return np.random.seed(x + int(time.time()))

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)
    # ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(BASE_DIR, 'models'))
    # print(sys.path)

    classes = ['not_fracture', 'fracture']
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat

    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace=True

    root = 'data/testdata/'
    # root = 'data/s3dis/stanford_indoor3d/'
    NUM_CLASSES = 2
    NUM_POINT = 4096
    BATCH_SIZE = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #####CHECKING WHAT THE DATA LOOKS LIKE############

    # example_data = np.load('data/s3dis/stanford_indoor3d/Area_1_conferenceRoom_1.npy')
    # print(example_data.shape)
    # print(example_data[0, :])
    # print(np.min(example_data[:, 3:6]))
    # print(np.max(example_data[:, 3:6]))

    # our_data = np.load('data/testdata/data_labelled_grid.npy')
    # print(our_data.shape) #662525
    # print(our_data[0, :])

    ###############OUR DATASET###############################################
    print("start loading training data ...")
    TRAIN_DATASET = FracDataset(data_root=root, split='train', num_point=NUM_POINT, block_size=4.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                    pin_memory=True, drop_last=True,
                                                    worker_init_fn = None)

    ###############ORIGINAL DATASET##########################################
    # print("start loading training data ...")
    # TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=5, block_size=1.0, sample_rate=1.0, transform=None)
    
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    #                                               pin_memory=True, drop_last=True)
    #                                             #   worker_init_fn=worker_init,
    #                                             #   persistent_workers=True)

    weights = torch.Tensor(TRAIN_DATASET.labelweights).to(device)
    print("The number of training data is: %d" % len(TRAIN_DATASET))

    ############### MODEL LOADING ###########################################
    '''MODEL LOADING'''
    MODEL = importlib.import_module("pointnet2_sem_seg")
    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    criterion = MODEL.get_loss().to(device)

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4)
    
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 10

    global_epoch = 0
    best_iou = 0

    ######################## TRAINING ###############################################

    for epoch in range(start_epoch, 32):

        '''Train on chopped scenes'''
        print('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, 32))
        lr = max(0.001 * (0.7 ** (epoch // 10)), LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss

            global_epoch += 1

        print('Training mean loss: %f' % (loss_sum / num_batches))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))


    # ###############TESTING DATALOADER ######################################

    # for i, (points, target) in enumerate(trainDataLoader):
    #     points = points.data.numpy()

    #     print("Points size", points.shape)
    #     print("Target size", target.size())

    #     if i == 0:
    #         break


if __name__ == '__main__':
    main()
