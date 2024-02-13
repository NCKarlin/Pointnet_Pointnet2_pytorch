import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import yaml
import importlib
from sklearn.metrics import confusion_matrix
from torchmetrics import Accuracy, Precision, Recall
from inference_analysis_utils import (
    load_specific_blocks,
    load_sample,
    normalize_rgb,
    create_rgb_list,
    pull_xyz_coords,
    create_marked_rgb_array,
    create_dataloading_lists,
    create_marked_point_sizes,
    generate_predictions, 
    create_and_save_figure
)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#! VARIABLES TO BE SET BEFORE RUNNING INFERENCE ANALYSIS SCRIPT
#########################################################################################
path_to_samples = '/home/innolidix/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/data/samples/train/sample_0'
path_to_outputs = '/home/innolidix/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/outputs'
figsavepath = '/home/innolidix/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/inference_figs'
date = '2024-02-06'
run_name = 'royal-blaze-628'
entire_sample = True
block_load_list = [15, 16, 17, 18]


# LOADING OF DATA TO BE INFERENCED
#########################################################################################
if entire_sample:
    points_ls, labels_ls = load_sample(path_to_samples)
else:
    points_ls, labels_ls = load_specific_blocks(path_to_samples, block_load_list)


# PRE-PROCESSING OF DATA TO BE INFERENCED - RGB-NORMALIZATION
#########################################################################################
if entire_sample:
    num_blocks = math.ceil(len(points_ls))
    points_arr = normalize_rgb(np.array(points_ls))
    points_data = points_arr.reshape(-1, points_arr.shape[2])
else:
    num_blocks = math.ceil(len(points_ls))
    points_arr = normalize_rgb(np.array(points_ls))
    points_data = points_arr.reshape(-1, points_arr.shape[2])

# LOADING OF PRE-TRAINED MODEL FOR INFERENCE
#########################################################################################
# Parsing needed hyperparameters from original config yaml file 
with open('/home/innolidix/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/conf/train/train_config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)
    model_name = config_data['hyperparams']['min_model']
    ncentroids = config_data['hyperparams']['ncentroids_min']
    radius = config_data['hyperparams']['radius_min']
    samples_around_centroid = config_data['hyperparams']['samples_around_centroid']
    sa_mlps = config_data['hyperparams']['sa_mlps_min']
    fp_mlps = config_data['hyperparams']['fp_mlps_min']
    loss_function = config_data['hyperparams']['loss_function']
    learning_rate = config_data['hyperparams']['learning_rate']
    decay_rate = np.float(config_data['hyperparams']['decay_rate'])
    batch_size = config_data['hyperparams']['batch_size']
    npoint = config_data['hyperparams']['npoint']
# Model instantiation
MODEL = importlib.import_module(model_name)
classifier = MODEL.get_model(2,
                            ncentroids,
                            radius, 
                            samples_around_centroid,
                            sa_mlps,
                            fp_mlps,
                            loss_function).to(DEVICE)
# Optimizer instantiation
optimizer = torch.optim.Adam(classifier.parameters(), 
                            lr=learning_rate,
                            betas=(0.9, 0.99),
                            eps=1e-08,
                            weight_decay=decay_rate)
# Model loading
model_path = os.path.join(path_to_outputs, date, run_name, 'models/best_model.pth')
best_checkpoint = torch.load(model_path)
classifier.load_state_dict(best_checkpoint['model_state_dict'])
optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
epoch = best_checkpoint['epoch']
class_avg_iou = best_checkpoint['class_avg_iou']


# DATASET AND DATA LOADER CREATION
#########################################################################################
test_points, test_labels = create_dataloading_lists(points_ls, 
                                                    labels_ls,
                                                    batch_size,
                                                    math.ceil(num_blocks/batch_size))


# MODEL INFERENCE
#########################################################################################
pred_labels = generate_predictions(classifier, 
                                   test_points,
                                   test_labels, 
                                   loss_function, 
                                   DEVICE)


# STATISTCS & ANALYSIS
#########################################################################################
# Data type and shape preparation for statistics and analysis
true_labels = np.array(labels_ls).ravel()
pred_labels_tensor = torch.tensor(pred_labels)
true_labels_tensor = torch.tensor(true_labels)
# 1. Accuracy
accuracy = Accuracy(task='binary', num_classes=2)
model_accuracy = accuracy(pred_labels_tensor, true_labels_tensor).numpy()
correct = np.sum((pred_labels == true_labels))
# 2. Precision
precision = Precision(task='binary', num_classes=2)
model_precision = precision(pred_labels_tensor, true_labels_tensor).numpy()
# 3. Recall 
recall = Recall(task='binary', num_classes=2) 
model_recall = recall(pred_labels_tensor, true_labels_tensor).numpy()
# 4. Confusion matrix 
conf_mat = confusion_matrix(pred_labels, true_labels)
tn, fp, fn, tp = conf_mat.ravel()
# Printing results
print(f'---------------------------------------------')
print('MODEL PERFORMANCE STATISTICS')
print(f'---------------------------------------------')
print('GENERAL STATISTICS')
print(f'Overall there are  {np.count_nonzero(true_labels)} fracture points in this sample.')
print(f'The model recognized {np.count_nonzero(pred_labels)}')
print(f'Out of {len(pred_labels)} points it predicted {correct} correctly.')
print(f'---------------------------------------------')
print('TORCHMETRICS STATISTICS')
print(f'Accuracy: {model_accuracy.round(4)}')
print(f'Precision: {model_precision.round(4)}')
print(f'Recall: {model_recall.round(4)}')
print(f'---------------------------------------------')
print('CONFUSION MATRIX')
print(f'TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}')
print(f'---------------------------------------------')
print('PER-CLASS STATISTICS')
print(f'Accuracy on fracture points: {np.round(tp/ np.count_nonzero(true_labels), 3)}')
print(f'Accuracy on non-fracture points: {np.round(tn/ (len(true_labels)-np.count_nonzero(true_labels)), 3)}')


# SAVING VISUALIZATION OF GROUND TRUTH & PREDICTIONS
#########################################################################################
# 1. PREDICTIONS
marked_preds_rgb_arr = create_marked_rgb_array(points_data, pred_labels)
preds_point_sizes = create_marked_point_sizes(pred_labels)
create_and_save_figure(points_data,
                       marked_preds_rgb_arr,
                       preds_point_sizes,
                       figsavepath,
                       figure_name='marked_predictions.png',
                       dpi=500)
# 2. GROUND TRUTH
marked_gt_rgb_arr = create_marked_rgb_array(points_data, true_labels)
gt_point_sizes = create_marked_point_sizes(true_labels)
create_and_save_figure(points_data,
                       marked_gt_rgb_arr,
                       gt_point_sizes,
                       figsavepath,
                       figure_name='marked_ground_truth.png',
                       dpi=500)

print('Figures successfully created!')