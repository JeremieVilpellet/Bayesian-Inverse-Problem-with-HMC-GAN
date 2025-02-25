import torch
import os
import glob
from torch.utils.data import Dataset, DataLoader

class DarcyDataset(Dataset):
    def __init__(self, data_path, num_files=300000):
        self.data_path = data_path
        self.files = glob.glob(os.path.join(data_path, "*.pt"))
        self.files = self.files[:num_files]
        
        if len(self.files) < num_files:
            raise ValueError("Pas assez de fichiers pour num_files demandé.")
            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])  # data shape: (4,50,50)
        
        # Scaling individuel pour chaque paramètre
        data[0] = torch.clamp(data[0], min=0.0, max=1.6)  # v1
        data[1] = torch.clamp(data[1], min=0.0, max=1.6)  # v2
        data[2] = torch.clamp(data[2], min=0.0, max=1.0)  # p
        data[3] = torch.clamp(data[3], min=0.0, max=2.5)  # k
        
    
        return data

def get_dataloader(data_path,
                   num_files=300000,
                   batch_size=64,
                   shuffle=True,
                   num_workers=1,
                   drop_last=True):
    dataset = DarcyDataset(data_path, num_files=num_files)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    return dataloader
    

 
        
