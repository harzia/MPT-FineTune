import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DictTensorDataset(Dataset):
    def __init__(self, inputs_dict, labels):
        self.inputs = inputs_dict
        self.labels = labels
        self.length = len(labels)

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.inputs.items()}, self.labels[index]

    def __len__(self):
        return self.length

def load_processed_split(data_dir, split_name, batch_size=128):
    path = os.path.join(data_dir, f"{split_name}_processed")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at {path}. Run preprocess_data.py first")

    print(f"Loading {split_name} data from {path}...")
    
    try:
        feats = np.load(os.path.join(path, "pf_features.npy"))
        vecs = np.load(os.path.join(path, "pf_vectors.npy"))
        mask = np.load(os.path.join(path, "pf_mask.npy"))
        labels = np.load(os.path.join(path, "labels.npy"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing .npy file in {path}. Did preprocessing finish successfully? Error: {e}")

    inputs = {
        'pf_features': torch.from_numpy(feats).float(),
        'pf_vectors': torch.from_numpy(vecs).float(),
        'pf_mask': torch.from_numpy(mask).float(),
    }
    targets = torch.from_numpy(labels).float()
    
    shuffle = (split_name == 'train')
    
    dataset = DictTensorDataset(inputs, targets)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def get_train_loader(data_dir, dataset_name, batch_size=128):
    return load_processed_split(data_dir, 'train', batch_size)

def get_test_loader(data_dir, dataset_name, batch_size=128):
    return load_processed_split(data_dir, 'test', batch_size)