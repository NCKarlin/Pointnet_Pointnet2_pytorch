"""
Author: Benny
Date: Nov 2019
"""
import os
from data_utils.S3DISDataLoader import FracDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import time
import hydra
import wandb
import omegaconf
import sys

log = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

classes = ['non_fracture', 'fracture']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def worker_init(x):
    return np.random.seed(x + int(time.time()))

@hydra.main(version_base="1.2", config_path= "conf", config_name="default_config.yaml")
def main(cfg):

    train_params = cfg.train.hyperparams
    log.info(cfg.train.hyperparams.comment)

    # setting up hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    OUTPUT_DIR = hydra_cfg['runtime']['output_dir']
    os.makedirs(OUTPUT_DIR + "/models/", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/figures/", exist_ok=True)

    #wandb setup
    myconfig = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 
    wandb.init(config = myconfig, project='FracRec', group = train_params.exp_group, notes=train_params.comment)

    DATA_ROOT = 'data/testdata/'
    NUM_CLASSES = 2
    NUM_POINT = train_params.npoint
    BATCH_SIZE = train_params.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")


    ########################### DATA LOADING ###########################
    print("start loading training data ...")
    TRAIN_DATASET = FracDataset(data_root=DATA_ROOT, split='train', num_point=NUM_POINT, block_size=4.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = FracDataset(data_root=DATA_ROOT, split='test', num_point=NUM_POINT, block_size=4.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).to(device)

    log.info("The number of training data is: %d" % len(TRAIN_DATASET))
    log.info("The number of test data is: %d" % len(TEST_DATASET))

    ########################### MODEL LOADING ###########################
    sys.path.append(os.path.join(BASE_DIR, 'models'))
    MODEL = importlib.import_module(train_params.model)
    # shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    criterion = MODEL.get_loss().to(device)
    classifier.apply(inplace_relu)

    ####################### INITIALIZING WEIGHTS #####################
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

    if train_params.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=train_params.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=train_params.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=train_params.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = train_params.step_size

    global_epoch = 0
    best_iou = 0

    ################################## TRAINING ###############################################
    for epoch in range(start_epoch, train_params.epoch):

        log.info('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, train_params.epoch))
        lr = max(train_params.learning_rate * (train_params.lr_decay ** (epoch // train_params.step_size)), LEARNING_RATE_CLIP)
        log.info('Learning rate:%f' % lr)

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
            wandb.log({"train loss": loss})

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log.info('Training mean loss: %f' % (loss_sum / num_batches))
        log.info('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            savepath = OUTPUT_DIR + "/models" + '/model.pth'
            log.info('Saving model at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        # '''Evaluate on chopped scenes'''
        # with torch.no_grad():
        #     num_batches = len(testDataLoader)
        #     total_correct = 0
        #     total_seen = 0
        #     loss_sum = 0
        #     labelweights = np.zeros(NUM_CLASSES)
        #     total_seen_class = [0 for _ in range(NUM_CLASSES)]
        #     total_correct_class = [0 for _ in range(NUM_CLASSES)]
        #     total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        #     classifier = classifier.eval()

        #     log.info('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
        #     for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
        #         points = points.data.numpy()
        #         points = torch.Tensor(points)
        #         points, target = points.float().to(device), target.long().to(device)
        #         points = points.transpose(2, 1)

        #         seg_pred, trans_feat = classifier(points)
        #         pred_val = seg_pred.contiguous().cpu().data.numpy()
        #         seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

        #         batch_label = target.cpu().data.numpy()
        #         target = target.view(-1, 1)[:, 0]
        #         loss = criterion(seg_pred, target, trans_feat, weights)
        #         loss_sum += loss
        #         pred_val = np.argmax(pred_val, 2)
        #         correct = np.sum((pred_val == batch_label))
        #         total_correct += correct
        #         total_seen += (BATCH_SIZE * NUM_POINT)
        #         tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
        #         labelweights += tmp

        #         for l in range(NUM_CLASSES):
        #             total_seen_class[l] += np.sum((batch_label == l))
        #             total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
        #             total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        #     labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        #     mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        #     log.info('eval mean loss: %f' % (loss_sum / float(num_batches)))
        #     log.info('eval point avg class IoU: %f' % (mIoU))
        #     log.info('eval point accuracy: %f' % (total_correct / float(total_seen)))
        #     log.info('eval point avg class acc: %f' % (
        #         np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

        #     iou_per_class_str = '------- IoU --------\n'
        #     for l in range(NUM_CLASSES):
        #         iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
        #             seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
        #             total_correct_class[l] / float(total_iou_deno_class[l]))

        #     log.info(iou_per_class_str)
        #     log.info('Eval mean loss: %f' % (loss_sum / num_batches))
        #     log.info('Eval accuracy: %f' % (total_correct / float(total_seen)))

        #     if mIoU >= best_iou:
        #         best_iou = mIoU
        #         log.info('Save model...')
        #         savepath = OUTPUT_DIR + "/models" '/best_model.pth'
        #         log.info('Saving at %s' % savepath)
        #         state = {
        #             'epoch': epoch,
        #             'class_avg_iou': mIoU,
        #             'model_state_dict': classifier.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }
        #         torch.save(state, savepath)
        #     log.info('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    main()
