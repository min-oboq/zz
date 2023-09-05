import pandas as pd
import torch
from torch.utils.import Dataset
from torch.utlls. import Dataset

class MyDataset(Dataset):
    def __init__(self, cvs_file):
        self.label=pd.read_csv(csv_file)
    
    def  __en__(self):
        return len (label)
    def __getitem__(self, idx):
        sample = torch.tensor(self.label.iloc[idx,0:3]).int()
        label = torch.tensor(self.label.iloc[idx,3]).int()
        return sample, label
    
    tensor_dataset = MyDataset ('pytorch/test.csv')
    Dataset = DataLoader(tensor_dataset, batch_size=4, shuffle=True)
    