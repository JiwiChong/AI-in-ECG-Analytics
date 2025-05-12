import torch
from scipy.signal import medfilt
from torch.utils.data import Dataset
import pywt
import numpy as np
import pandas as pd


class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def save_checkpoint(save_path, model, best_val_acc, loss):
    if save_path == None:
        return

    loss_txt = 'val_loss' 
    state_dict = {'model_state_dict': model.state_dict(),
                  loss_txt: loss,
                  'val_metric': best_val_acc}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    return state_dict
    

def load_and_prepare_data(args):
    """Load and prepare PTB-DB dataset"""

    if args.data == 'ptb_ecg':
        normal_df = pd.read_csv(f'{args.data_dir}/ptbdb_normal.csv', header=None)
        abnormal_df = pd.read_csv(f'{args.data_dir}/ptbdb_abnormal.csv', header=None)
        combined_data = pd.concat([normal_df, abnormal_df], axis=0, ignore_index=True)
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    elif args.data == 'mit-bih':
        combined_data = pd.read_csv(f'{args.data_dir}/entire_df.csv', index_col=0)

    X_data = combined_data.iloc[:, :-1].values
    y_data = combined_data.iloc[:, -1].values
    
    print("\n Pre Class distribution:")
    unique, counts = np.unique(y_data, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples ({count/len(y_data)*100:.2f}%)")
    
    return X_data, y_data
    

def wavelet_denoising(x):
    coeffs = pywt.wavedec(x, 'db4', level=4)
    threshold = np.std(coeffs[-1])/2
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised = pywt.waverec(coeffs, 'db4')
    return denoised

def median_filtering(x_denoised):
    return x_denoised - medfilt(x_denoised, kernel_size=51)

def normalization(x_corrected):
    return np.clip(((x_corrected - np.mean(x_corrected))/np.std(x_corrected) + 10e-8), -5, 5)
