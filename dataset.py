import numpy as np
import glob
import scipy.io as sio
import torch
import os
from torch.utils.data import Dataset, DataLoader


def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y


class MyCSIDataset(Dataset):
    def __init__(self, data_path, label_path, mode='flat', transform=None, time_steps=100):
        self.data = np.load(data_path)   # shape: (N, 100, 4004) or (N, 4004)
        self.labels = np.load(label_path)
        self.transform = transform
        self.mode = mode
        self.time_steps = time_steps
        self.max_features = 4096

        if mode == 'cnn':
            # Frame-level classification (1 frame per sample)
            if self.data.shape[1] != self.max_features:
                self.data = np.pad(self.data, ((0, 0), (0, self.max_features - self.data.shape[1])))
            self.data = self.data.reshape(-1, 1, 64, 64)  # (N, 1, 64, 64)

        elif mode == 'snn':
            # Segment-level classification for SNNs: (N, T, 4096) → (N, T, 1, 64, 64)
            if self.data.shape[2] != self.max_features:
                self.data = np.pad(self.data, ((0, 0), (0, 0), (0, self.max_features - self.data.shape[2])))
            self.data = self.data.reshape(-1, self.time_steps, 1, 64, 64)

        elif mode == 'rnn':
            # Segment-level for GRU: shape = (N, T, 4096)
            if self.data.shape[2] != self.max_features:
                self.data = np.pad(self.data, ((0, 0), (0, 0), (0, self.max_features - self.data.shape[2])))

        elif mode == 'flat':
            # Keep as is (e.g., (N, 4004) or (N, 4096))
            pass

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# class MyCSIDataset(Dataset):
#     def __init__(self, data_path, label_path, mode='flat', transform=None):
#         self.data = np.load(data_path)
#         self.labels = np.load(label_path)
#         self.transform = transform

#         # Reshape data based on model type
#         if mode == 'cnn':
#             if self.data.shape[1] != 4096:
#                 self.data = np.pad(self.data, ((0, 0), (0, 4096 - self.data.shape[1])))
#             self.data = self.data.reshape(-1, 1, 64, 64)
#         elif mode == 'rnn':
#             self.data = self.data.reshape(-1, 1001, 4)
#         elif mode == 'flat':
#             pass  # (N, 4004)
#         else:
#             raise ValueError(f"Unknown model_type: {mode}")

#         self.data = torch.tensor(self.data, dtype=torch.float32)
#         self.labels = torch.tensor(self.labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         x = self.data[idx]
#         y = self.labels[idx]
#         if self.transform:
#             x = self.transform(x)
#         return x, y

class OTFSegmentDataset(Dataset):
    def __init__(self, frame_data_path, frame_label_path,
                 segment_size=100, step_size=50, mode='rnn', transform=None,
                 max_features=4096):
        """
        On-the-fly segment dataset from per-frame data.

        Args:
            frame_data_path: path to X_frame_*.npy
            frame_label_path: path to y_frame_*.npy
            segment_size: number of frames per segment
            step_size: stride between segments
            mode: 'rnn' | 'snn' | 'cnn+gru'
            transform: optional transform function
        """
        self.X = np.load(frame_data_path)  # shape: (N, 4004/4096)
        self.y = np.load(frame_label_path) # shape: (N,)
        self.segment_size = segment_size
        self.step_size = step_size
        self.mode = mode
        self.transform = transform
        self.max_features = max_features

        if self.X.shape[1] < max_features:
            pad_width = max_features - self.X.shape[1]
            self.X = np.pad(self.X, ((0, 0), (0, pad_width)))

        # Build segment index lookup (start positions)
        self.indices = []
        for i in range(0, len(self.X) - segment_size + 1, step_size):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.segment_size
        segment = self.X[start:end]  # shape: (segment_size, 4096)
        label_segment = self.y[start:end]

        # Majority label (simple heuristic)
        label = int(np.bincount(label_segment).argmax())

        if self.mode == 'rnn':
            x = segment  # (segment_size, 4096)

        elif self.mode in ['snn', 'cnn+gru']:
            # Reshape to (segment_size, 1, 64, 64)
            x = segment.reshape(self.segment_size, 1, 64, 64)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.transform:
            x = self.transform(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    

class OTFSegmentPerFileDataset(Dataset):
    def __init__(self, folder, segment_size=100, step_size=50, mode='rnn', transform=None, max_features=4096):
        """
        On-the-fly segmentation from per-trial .npy files.

        Args:
            folder: directory containing trial .npy files (e.g., Sitting_1.npy)
            segment_size: frames per segment
            step_size: stride between segments
            mode: 'rnn' | 'snn' | 'cnn+gru'
            transform: optional data transform
        """
        self.segment_size = segment_size
        self.step_size = step_size
        self.mode = mode
        self.transform = transform
        self.max_features = max_features

        self.segments = []  # List of (file_path, start_idx, label)

        # Class label mapping from file names
        self.label_map = {}  # e.g., {"Walking": 4}
        label_id = 0

        # for file in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        #     filename = os.path.basename(file)
        #     activity = filename.split('_')[0]  # e.g., "Sitting_1.npy" → "Sitting"
        #     if activity not in self.label_map:
        #         self.label_map[activity] = label_id
        #         label_id += 1
        #     label = self.label_map[activity]

        #     data = np.load(file)  # (T, 4004) or fewer
        #     if len(data.shape) == 1:
        #         data = data[np.newaxis, :]

        #     # Pad if needed
        #     if data.shape[1] < max_features:
        #         pad_width = max_features - data.shape[1]
        #         data = np.pad(data, ((0, 0), (0, pad_width)))

        #     # Store all valid start indices within this file
        #     for start in range(0, len(data) - segment_size + 1, step_size):
        #         self.segments.append((file, start, label))

        self.cache = {}  # NEW: stores preloaded data

        for file in sorted(glob.glob(os.path.join(folder, "*.npy"))):
            filename = os.path.basename(file)
            activity = filename.split('_')[0]
            if activity not in self.label_map:
                self.label_map[activity] = label_id
                label_id += 1
            label = self.label_map[activity]

            # Load and pad now
            data = np.load(file)
            if len(data.shape) == 1:
                data = data[np.newaxis, :]
            if data.shape[1] < max_features:
                pad_width = max_features - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)))

            self.cache[file] = data  # ✅ preload into memory

            for start in range(0, len(data) - segment_size + 1, step_size):
                self.segments.append((file, start, label))

        print(f"🔄 Loaded {len(self.segments)} segments from {len(self.label_map)} classes.")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        file, start, label = self.segments[idx]
        data = self.cache[file]
        segment = data[start:start + self.segment_size]

        if self.mode == 'rnn':
            x = segment
        elif self.mode in ['snn', 'cnn+gru']:
            x = segment.reshape(self.segment_size, 1, 64, 64)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.transform:
            x = self.transform(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    # def __getitem__(self, idx):
    #     file, start, label = self.segments[idx]
    #     data = np.load(file)
    #     if len(data.shape) == 1:
    #         data = data[np.newaxis, :]
    #     if data.shape[1] < self.max_features:
    #         pad_width = self.max_features - data.shape[1]
    #         data = np.pad(data, ((0, 0), (0, pad_width)))

    #     segment = data[start:start + self.segment_size]  # (T, 4096)

    #     if self.mode == 'rnn':
    #         x = segment  # (T, 4096)

    #     elif self.mode in ['snn', 'cnn+gru']:
    #         x = segment.reshape(self.segment_size, 1, 64, 64)  # (T, 1, 64, 64)

    #     else:
    #         raise ValueError(f"Unsupported mode: {self.mode}")

    #     if self.transform:
    #         x = self.transform(x)

    #     return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)