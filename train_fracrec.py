"""
Original:
Author: Benny
Date: Nov 2019

Adjusted by:
Authors: Kristin Remmelgas & Niklas Karlin
Date: Sep 2023
"""
import os
from data_utils.S3DISDataLoader import FracDataset
import torch
import logging
import sys
import importlib
import provider
import numpy as np
import time
import hydra
import wandb
import omegaconf
import sys
import math
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, roc_auc_score, RocCurveDisplay
log = logging.getLogger(__name__)

#PARAMETER SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join("data", "testdata", "")
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    torch.cuda.empty_cache() #for memory cleaning before every run
    train_params = cfg.train.hyperparams
    log.info(f"Using device: {DEVICE}")
    log.info(cfg.train.hyperparams.comment)

    # setting up hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    OUTPUT_DIR = hydra_cfg['runtime']['output_dir']
    os.makedirs(os.path.join(OUTPUT_DIR, "models", ""), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures" ""), exist_ok=True)

    #wandb setup
    myconfig = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 
    wandb.init(config = myconfig, 
               project='FracRec', 
               entity='innolidix', 
               group = train_params.exp_group, 
               notes=train_params.comment,
               tags=[train_params.tag])

    ########################### DATA LOADING ###########################
    print("Start loading training data ...")
    TRAIN_DATASET = FracDataset(data_root=DATA_ROOT, split='train', num_point=train_params.npoint, block_size=train_params.block_size, sample_rate=train_params.sample_rate, transform=None)
    print("Start loading test data ...")
    TEST_DATASET = FracDataset(data_root=DATA_ROOT, split='train', num_point=4096, block_size=train_params.block_size, sample_rate=train_params.sample_rate, transform=None)

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

    classifier = MODEL.get_model(NUM_CLASSES, 
                                 train_params.ncentroids,
                                 train_params.radius,
                                 train_params.samples_around_centroid,
                                 train_params.sa_mlps,
                                 train_params.fp_mlps,
                                 train_params.dropout,
                                 train_params.dropout_prob).to(DEVICE)
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
    n_steps_per_epoch = math.ceil(len(trainDataLoader.dataset) / train_params.batch_size)

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
            if train_params.additional_rotation:
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().to(DEVICE), target.long().to(DEVICE)
            points = points.transpose(2, 1)

            print(f"------ Training for batch {i} ------") #print check for CUDA error: device-side assert triggered
            seg_pred, trans_feat, probs = classifier(points, 
                                                     train_params.loss_function,
                                                     train_params.dropout)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0].to(DEVICE)
            probs = probs.contiguous().view(-1, NUM_CLASSES)
            
            # Preparing and running loss depending on chosen loss function
            if train_params.loss_function == "BCE-Loss":
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)[:,0]
                loss_weights = torch.where(target==0.0, weights[0], weights[1])  
                batch_loss = criterion(train_params.loss_function, seg_pred, target.float(), trans_feat, loss_weights)
                #pred_choice = seg_pred.cpu().data.numpy()
                # Probabilities as pred_choice
                pred_choice = probs.contiguous().view(-1, NUM_CLASSES)[:,0]
            elif train_params.loss_function == "CE-Loss" or train_params.loss_function == "NLL-Loss" :
                batch_loss = criterion(train_params.loss_function, seg_pred, target, trans_feat, weights)  
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            
            batch_loss.backward()
            optimizer.step()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (train_params.batch_size * train_params.npoint)
            train_losses_epoch.append(batch_loss.item())

            if i % 10 == 9:  # print and log average training loss every 10 batches
                avg_trainloss_10batches = sum(train_losses_epoch[-10:]) / 10
                log.info("[%d, %5d] train loss: %.3f" % (epoch + 1, i + 1, avg_trainloss_10batches))

            metrics = {"train/train_loss": batch_loss}
            if i + 1 < n_steps_per_epoch:
                wandb.log(metrics)

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

            y_pred, y_true, y_true, y_probs_pos = [], [], [], []
            tp_all, fp_all, tn_all, fn_all = [], [], [], [] #for average values

            log.info('********** Epoch %d/%s EVALUATION **********' % (epoch + 1, train_params.epoch))
            for i, (points, target) in enumerate(testDataLoader):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(DEVICE), target.long().to(DEVICE)
                points = points.transpose(2, 1)

                print(f"----- Evaluating for batch {i} ------") #print check for CUDA error: device-side assert triggered
                seg_pred, trans_feat, probs = classifier(points, 
                                                         train_params.loss_function,
                                                         train_params.dropout)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                prob_val = probs.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0].to(DEVICE)
                
                # Preparing and running loss depending on chosen loss function
                if train_params.loss_function == "BCE-Loss":
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)[:,0]
                    loss_weights = torch.where(target==0.0, weights[0], weights[1])   
                    loss = criterion(train_params.loss_function, seg_pred, target.float(), trans_feat, loss_weights)
                    # probabilities as pred_val
                    pre_pred_val = probs.contiguous().view(-1, NUM_CLASSES)[:,0].cpu().data.numpy() #one dimensional
                    pred_val = np.zeros_like(pre_pred_val)
                    pred_val[pre_pred_val > 0.5] = 1
                elif train_params.loss_function == "CE-Loss" or train_params.loss_function == "NLL-Loss" :
                    loss = criterion(train_params.loss_function, seg_pred, target, trans_feat, weights)
                    pred_val = np.argmax(pred_val, 2) #two dimensional
                else:
                    print("Loss function has not been specified sufficiently for evaluation script.")
                
                loss_sum += loss
                val_losses_epoch.append(loss.item())
                prob_val_pos = prob_val[:, :, 1]
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (train_params.batch_size * train_params.npoint)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                #Keeping track of overall true values and predicted values for the confusion matrix and f1 score
                y_pred.extend(pred_val.flatten().tolist())
                y_true.extend(batch_label.flatten().tolist())
                y_probs_pos.extend(prob_val_pos.flatten().tolist())

                # print average validation loss for every 10 batches (but wandb logging only the batch mean value)
                if i % 10 == 9:  
                    avg_valloss_10batches = sum(val_losses_epoch[-10:]) / 10
                    log.info("[%d, %5d] validation loss: %.3f" % (epoch + 1, i + 1,  avg_valloss_10batches))

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
            
            # Build confusion matrix
            # TODO: Adjust the structure, so that it tracks it entirely and calculates and displays averages
            cf_matrix = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cf_matrix.ravel()
            # Saving all for overall average values
            tn_all.append(tn)
            fp_all.append(fp)
            fn_all.append(fn)
            tp_all.append(tp)
            log.info('Confusion matrix - TP: %.3f, FP: %.3f, FN: %.3f, TN: %.3f.' % (tp, fp, fn, tn))
            confusionMatrixPlot(y_true, y_pred, OUTPUT_DIR)

            #Get F1 score
            f1score = f1_score(y_true, y_pred, average='weighted')
            log.info('The F1 score: %.3f.' % (f1score))

            #Making the ROC curve and finding the AUC
            aucscore = roc_auc_score(y_true, y_probs_pos)
            log.info('The ROC AUC score: %.3f.' % (aucscore))
            fpr, tpr, thresholds = roc_curve(y_true, y_probs_pos)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucscore, estimator_name='example estimator').plot()
            plt.savefig(os.path.join(OUTPUT_DIR, "figures", "roc_curve_plot.jpg"))

            #Other eval metrics
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            acc_class_mean = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
            log.info('Eval mean loss per epoch: %f' % (loss_sum / float(num_batches)))
            log.info('Eval mean accuracy per epoch: %f' % (total_correct / float(total_seen)))
            log.info('Eval class mean IoU: %f' % (mIoU))
            log.info('Eval class mean accuracy: %f' % (acc_class_mean))

            log.info('------- IoU per class --------')
            for l in range(NUM_CLASSES):
                log.info('Class %s weight: %.3f, IoU: %.3f' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1], total_correct_class[l] / float(total_iou_deno_class[l])))

            # wandb logging all validation metrics once per epoch
            val_metrics = {"val/validation_loss": np.mean(np.array(val_losses_epoch)),
                            "val/validation_accuracy": total_correct / float(total_seen),
                            "val/mIoU": mIoU,
                            "val/mAcc": acc_class_mean,
                            "val/conf_mat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=["Non-fracture", "Fracture"]),
                            "val/tp": tp,
                            "val/fp": fp,
                            "val/fn": fn,
                            "val/tn": tn,
                            "val/avg_tp": np.sum(tp_all)/ epoch, 
                            "val/avg_fp": np.sum(fp_all)/ epoch,
                            "val/avg_fn": np.sum(fn_all)/ epoch,
                            "val/avg_tn": np.sum(tn_all)/ epoch,
                            "val/f1_score": f1score,
                            "val/auc_score": aucscore,}
            wandb.log({**metrics, **val_metrics})

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

    wandb.finish()


def confusionMatrixPlot(y_true, y_pred, OUTPUT_DIR):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=np.array(["Not-fracture", "Fracture"]), cmap=plt.cm.Blues, colorbar = False)
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "confusion_matrix.jpg"))
    return


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
