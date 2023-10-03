import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from models.pointnet2_utils import pc_normalize

# DATASET CREATION CLASS FOR OUR MODEL ##################################################
class FracDataset(Dataset):
    def __init__(self, 
                 data_root,
                 split='train', 
                 num_point=4096, 
                 block_size=1.0, 
                 sample_rate=1.0,
                 transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        #taking files meant for training or testing depending on split argument (checking for 'train' or 'test' in file name)
        rooms = sorted(os.listdir(data_root))
        if split == 'train':
            #list of filenames to use
            rooms_split = [room for room in rooms if not 'test' in room]
        else:
            rooms_split = [room for room in rooms if 'test' in room]

        #placeholders for aggregate data from all rooms combined
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = [] #list of the number of points in each file
        labelweights = np.zeros(2) #total number of points belonging to each label

        #looping over room files to combine data into one dataset
        for room_name in tqdm(rooms_split, total=len(rooms_split)):

            #getting specific room data from file
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N

            tmp, _ = np.histogram(labels, range(3)) #list of how many points are of each label in this room/file [3637994, 11176]
            labelweights += tmp #adding to total over all files (same shape)

            #getting min and max of all coordinates in this room/file
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

            #appending this file data to combined dataset
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
            #TODO: For now insert the print with the target counts of points etc. here

        # OWN LABELWEIGHTING
        labelweights = labelweights.astype(np.float32)
        pct_labelweights = labelweights / np.sum(labelweights)
        # w = N (# of all points) / [2 * N_j] (# of points of class j)
        self.labelweights = np.sum(labelweights) / (2 * labelweights)
        # "just" Inverse weighing of labels
        #self.labelweights = np.sum(labelweights) / labelweights

        sample_prob = num_point_all / np.sum(num_point_all) #list of probabilities of each file points being chosen from total points (file weights)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point) #nr of blocks 890 (total nr of points times sample rate (1.0) divided by number of points in block (4096))
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs) #list of indexes showing which room/file each block is from
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx] #index of the room/file where the block data is from
        points = self.room_points[room_idx]   # room points N * 6
        labels = self.room_labels[room_idx]   # room labels N
        coord_min = self.room_coord_min[room_idx]
        coord_max = self.room_coord_max[room_idx]
        N_points = points.shape[0] #number of room points

        # while (True):
        #     center = points[np.random.choice(N_points)][:3] #taking a completely random point in the room as centre
        #     block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0] #half a blocksize to negative side of centre
        #     block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0] #half a blocksize to positive side of centre
        #     #getting indexes of the points that are within the block xy range
        #     point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
        #     # print(point_idxs.size)
        #     #if there are less than 1024 points in the block then choose a different centre
        #     if point_idxs.size > 1024:
        #         break
            
        # (Sub-)Sampling with true mean of the input point cloud
        center = np.mean(points, axis=0)[:3]
        block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
        point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
        #2048
        if point_idxs.size < 2048:
            # Creating off centers
            off_centers = np.array([
                [0.25 * coord_max[0], 0.25 * coord_max[1], center[2]], 
                [0.25 * coord_max[0], 0.5 * coord_max[1], center[2]], 
                [0.25 * coord_max[0], 0.75 * coord_max[1], center[2]], 
                [0.5 * coord_max[0], 0.25 * coord_max[1], center[2]],  
                [0.5 * coord_max[0], 0.75 * coord_max[1], center[2]], 
                [0.75 * coord_max[0], 0.25 * coord_max[1], center[2]], 
                [0.75 * coord_max[0], 0.5 * coord_max[1], center[2]], 
                [0.75 * coord_max[0], 0.75 * coord_max[1], center[2]], 
            ])
            # Creation of list to pick newly sampled points from
            points_idxs_list = []
            points_len_list = []
            for off_center in off_centers:
                block_min = off_center - [self.block_size / 2.0, self.block_size / 2.0, 0]
                block_max = off_center + [self.block_size / 2.0, self.block_size / 2.0, 0]
                point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                points_idxs_list.append(point_idxs)
                points_len_list.append(len(point_idxs))
            # Checking for any center with more than 1024 points within the block
            if any(point_len>2048 for point_len in points_len_list):
                # Grabbing center with most points
                points_len_list_max_idx = np.argmax(points_len_list)
                point_idxs = points_idxs_list[points_len_list_max_idx]
                center = off_centers[points_len_list_max_idx]
            else:
                # Random point selection as before for now
                while (True):
                    center = points[np.random.choice(N_points)][:3] #taking a completely random point in the room as centre
                    block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0] #half a blocksize to negative side of centre
                    block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0] #half a blocksize to positive side of centre
                    #getting indexes of the points that are within the block xy range
                    point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                    #if there are less than 1024 points in the block then choose a different center
                    if point_idxs.size > 2048:
                        break

        #sampling exactly num_point (4096) points from the block points. If there are less points then some appear multiple times.
        np.random.seed(42) # random seed for replicating random selection
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            num_duplicate_points = 0
            print("No duplicate points when sampling blocks.")
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
            num_duplicate_points = self.num_point - point_idxs.size
            print(f"Number of duplicate points: {num_duplicate_points}")
        
        # normalize
        selected_points = points[selected_point_idxs, :]  # resampled points in the block: num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9

        #the x and y coordinate is shifted according to the centre of the block
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]

        #normalizing the rgb values
        selected_points[:, 3:6] /= 255.0

        #the new last 3 columns are the xyz columns divided by the corresponding max room coordinate
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]

        #putting it all together
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        #additional transforms if there are any
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
            
        return current_points, current_labels


    def __len__(self):
        return len(self.room_idxs)


# OLD NON-USED DATASET CREATION CLASSES #################################################
class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point: #>4096
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else: #1024-4096
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

# OLD NON-USED MAIN METHOD FOR DATSET CREATION ##########################################
if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()