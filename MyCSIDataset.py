import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class MyCSIDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='flat', normalize=True, max_len=None):
        """
        Args:
            root_dir: Folder containing CSV files
            transform: Optional transform to apply to each sample
            mode: One of ['flat', 'rnn', 'cnn'] → defines reshape strategy
            normalize: Whether to min-max normalize each sample
            max_len: Optional, for limiting number of packets per file
        """
        self.data = []
        self.labels = []
        self.label_map = {}
        self.transform = transform
        self.mode = mode
        self.normalize = normalize

        all_files = glob.glob(os.path.join(root_dir, '*.csv'))
        label_idx = 0

        for f in all_files:
            # Extract label from filename
            label_name = os.path.basename(f).split('_')[0]
            if label_name not in self.label_map:
                self.label_map[label_name] = label_idx
                label_idx += 1
            label_id = self.label_map[label_name]

            # Load and optionally limit number of packets
            arr = np.loadtxt(f, delimiter=';')  # shape: (N, 4004)
            if max_len:
                arr = arr[:max_len]
            for row in arr:
                if self.normalize:
                    row = (row - np.min(row)) / (np.max(row) - np.min(row) + 1e-8)

                # Reshape depending on the model
                if self.mode == 'flat':
                    x = row  # shape: (4004,)
                elif self.mode == 'rnn':
                    x = row.reshape(1001, 4)  # shape: (time_steps, features)
                elif self.mode == 'cnn':
                    padded = np.pad(row, (0, 4096 - 4004))  # pad to square
                    x = padded.reshape(1, 64, 64)  # shape: (1, 64, 64)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                self.data.append(torch.tensor(x, dtype=torch.float32))
                self.labels.append(label_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
