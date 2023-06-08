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
import matplotlib.pyplot as plt
import shutil
log = logging.getLogger(__name__)

#PARAMETER SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join("data", "testdata", "")
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {DEVICE}")

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
    os.makedirs(os.path.join(OUTPUT_DIR, "models", ""), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures" ""), exist_ok=True)

    #wandb setup
    # myconfig = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 
    # wandb.init(config = myconfig, project='FracRec', group = train_params.exp_group, notes=train_params.comment)

    ########################### DATA LOADING ###########################
    print("Start loading training data ...")
    TRAIN_DATASET = FracDataset(data_root=DATA_ROOT, split='train', num_point=train_params.npoint, block_size=4.0, sample_rate=1.0, transform=None)
    print("Start loading test data ...")
    TEST_DATASET = FracDataset(data_root=DATA_ROOT, split='test', num_point=train_params.npoint, block_size=4.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=train_params.batch_size, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=train_params.batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).to(DEVICE)

    log.info("The number of training data is: %d" % len(TRAIN_DATASET))
    log.info("The number of test data is: %d" % len(TEST_DATASET))

    ########################### MODEL LOADING ###########################
    sys.path.append(os.path.join(BASE_DIR, 'models'))
    MODEL = importlib.import_module(train_params.model)
    shutil.copy('models/%s.py' % train_params.model, os.path.join(OUTPUT_DIR, "models", ""))
    shutil.copy(os.path.join("models", "pointnet2_utils.py"), os.path.join(OUTPUT_DIR, "models", ""))

    classifier = MODEL.get_model(NUM_CLASSES).to(DEVICE)
    criterion = MODEL.get_loss().to(DEVICE)
    classifier.apply(inplace_relu)

    ####################### INITIALIZING WEIGHTS AND OPTIMIZER #####################
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

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

    start_epoch = 0
    best_iou = 0
    train_losses_total, val_losses_total = [], []

    for epoch in range(start_epoch, train_params.epoch):

        log.info('********** Epoch %d/%s TRAINING **********' % (epoch + 1, train_params.epoch))
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
        total_correct, total_seen = 0, 0
        train_losses_epoch, val_losses_epoch = [], []
        classifier = classifier.train()

        ################################## TRAINING ###############################################
        for i, (points, target) in enumerate(trainDataLoader):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().to(DEVICE), target.long().to(DEVICE)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            batch_loss = criterion(seg_pred, target, trans_feat, weights)
            batch_loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (train_params.batch_size * train_params.npoint)
            train_losses_epoch.append(batch_loss.item())

            if i % 10 == 9:  # print and log average training loss every 10 batches
                avg_trainloss_10batches = sum(train_losses_epoch[-10:]) / 10
                log.info("[%d, %5d] train loss: %.3f" % (epoch + 1, i + 1, avg_trainloss_10batches))
                # wandb.log({"train loss": avg_trainloss_10batches})

        log.info('Training mean loss per epoch: %f' % (sum(train_losses_epoch) / num_batches))
        log.info('Training accuracy per epoch: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            savepath = os.path.join(OUTPUT_DIR, "models", "model.pth")
            state = {'epoch': epoch, 'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}
            torch.save(state, savepath)

        ################################## EVALUATION ###############################################
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct, total_seen, loss_sum  = 0, 0, 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class, total_correct_class, total_iou_deno_class = [0 for _ in range(NUM_CLASSES)], [0 for _ in range(NUM_CLASSES)], [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log.info('********** Epoch %d/%s EVALUATION **********' % (epoch + 1, train_params.epoch))
            for i, (points, target) in enumerate(testDataLoader):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(DEVICE), target.long().to(DEVICE)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                val_losses_epoch.append(loss.item())
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (train_params.batch_size * train_params.npoint)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                # print average validation loss for every 10 batches (but wandb logging only the batch mean value)
                if i % 10 == 9:  
                    avg_valloss_10batches = sum(val_losses_epoch[-10:]) / 10
                    log.info("[%d, %5d] validation loss: %.3f" % (epoch + 1, i + 1,  avg_valloss_10batches))

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log.info('Eval mean loss per epoch: %f' % (loss_sum / float(num_batches)))
            log.info('Eval point accuracy per epoch: %f' % (total_correct / float(total_seen)))
            log.info('Eval point avg class IoU: %f' % (mIoU))
            log.info('Eval point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))

            log.info('------- IoU per class --------')
            for l in range(NUM_CLASSES):
                log.info('class %s weight: %.3f, IoU: %.3f' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l])))

            #wandb logging validation loss once per epoch
            # wandb.log({"validation loss": np.mean(np.array(val_losses_epoch))})

            #Saving the best model
            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = os.path.join(OUTPUT_DIR, "models", "best_model.pth")
                state = {'epoch': epoch, 'class_avg_iou': mIoU, 'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}
                torch.save(state, savepath)
                log.info('Saving the best model at %s' % savepath)
            log.info('Best mIoU: %f' % best_iou)

        train_losses_total.extend(train_losses_epoch)
        val_losses_total.extend(val_losses_epoch)

    #Saving losses to file
    np.save(os.path.join(OUTPUT_DIR, "figures", "train_losses.npy"), train_losses_total)
    np.save(os.path.join(OUTPUT_DIR, "figures", "val_losses.npy"), val_losses_total)

    #Making and saving loss plot
    lossPlot(np.array(train_losses_total), np.array(val_losses_total), train_params.epoch, OUTPUT_DIR)


def lossPlot(train_losses_total, val_losses_total, epochs, OUTPUT_DIR):
    epoch_mean_val_losses = np.mean(val_losses_total.reshape(-1, int(len(val_losses_total) / epochs)), axis=1)

    x_train = np.arange(1, len(train_losses_total)+1)
    x_val = np.arange(1, epochs+1)*len(train_losses_total)/epochs

    plt.figure(figsize=(9, 9))
    plt.plot(x_train, train_losses_total, 'r', marker='o', linestyle='-', label="Training Error")
    plt.plot(x_val, epoch_mean_val_losses, 'b', marker='o', linestyle='-', label="Validation Error")
    plt.grid()
    plt.legend(fontsize=20)
    plt.xlabel("Train step", fontsize=20)
    plt.ylabel("Error", fontsize=20)
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "basic_loss_plot.jpg"))
    return

if __name__ == '__main__':
    main()
