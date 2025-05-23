import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
from models.ecg_model import CNN1D_Att
from loss_functions import FocalLoss
from utils import load_and_prepare_data, wavelet_denoising, median_filtering, normalization, save_checkpoint, ECGDataset
from class_balancer import smote_class_balancing
from torch.utils.data import DataLoader
import argparse
import gc
import os
import pickle


def run(args, model, train_loader, val_loader,criterion, optimizer, scheduler):
    # Training loop
    epochs = []                
    epoch_train_loss = []
    epoch_valid_loss = []

    if args.ensemble:
        model_type, ensemble = 'ensemble', []
    else:
        model_type, ensemble = 'non_ensemble', None

    best_loss = np.inf
    best_path = f'./saved_models/{args.model_name}/{args.data}/{model_type}/run_{args.run_num}/best_model_num_{args.run_num}.pt' 

    print('------------  Training started! --------------')
    num_epochs = args.epochs
    for epoch in tqdm(range(num_epochs)):
        model.train()

        b_train_loss = []
        train_correct, train_total = 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(inputs)

            new_label_tensor = np.zeros((len(labels), args.classes))
            for i, e in enumerate(list(labels)):
                new_label_tensor[i,int(e)] = 1
            labels__ = torch.tensor(new_label_tensor, dtype=torch.float32)
        
            loss = criterion(outputs, labels__.to(device))
            b_train_loss.append(loss.item())

            pred_y_label = torch.argmax(outputs.detach().cpu(), dim=1)
            true_y_label = labels.detach().cpu().squeeze(1)
            train_correct += (pred_y_label == true_y_label).sum().item()

            train_total += labels.size(0)

            del inputs
            del labels
            torch.cuda.empty_cache()
            gc.collect()
            
            loss.backward()
            optimizer.step()
        
        epoch_train_loss.append(np.mean(b_train_loss))
        print('Epoch: {}'.format(epoch+1))
        print('Training Loss: {}'.format(np.mean(b_train_loss)))
        print('Training Accuracy {}'.format(round(train_correct / train_total, 3)))

        model.eval()

        with torch.no_grad():
            b_valid_loss = []
            # b_valid_y = []
            # b_valid_y_hat = []
            val_correct, val_total = 0, 0
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)

                outputs = model(inputs)
                new_label_tensor = np.zeros((len(labels), args.classes))
                for i, e in enumerate(list(labels)):
                    new_label_tensor[i,int(e)] = 1
                labels__ = torch.tensor(new_label_tensor, dtype=torch.float32)
                val_loss = criterion(outputs, labels__.to(device))

                b_valid_loss.append(val_loss.item())

                pred_y_label = torch.argmax(outputs.detach().cpu(), dim=1)
                true_y_label = labels.detach().cpu().squeeze(1)
                val_correct += (pred_y_label == true_y_label).sum().item()

                val_total += labels.size(0)

            print('Validation Loss {}'.format(np.mean(b_valid_loss)))
            print('Validation Accuracy {}'.format(round(val_correct / val_total, 3)))
            print('-' * 40)
            epoch_valid_loss.append(np.mean(b_valid_loss))
            
            if np.mean(b_valid_loss) < best_loss:
                best_loss = np.mean(b_valid_loss)
                results_dict = save_checkpoint(best_path, model, round(val_correct / val_total, 3), np.mean(b_valid_loss))
                if ensemble is None:
                    pass
                else:
                    ensemble.append(results_dict)

        epochs.append(epoch+1)
    print("Training complete!")
    return ensemble

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECG Classification')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model_name', type=str, help='model')
    parser.add_argument('--data', type=str, help='dataset')
    parser.add_argument('--ensemble', type=bool, default=False, help='ensemble or not')
    parser.add_argument('--loss_func', type=str, default='cross_ent',help='Loss function')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help='learning rate')
    parser.add_argument('--classes', type=int, help='Num of classes')
    parser.add_argument('--class_balance', type=bool, default=False, help='whether to balance labels or not')
    parser.add_argument('--scaler', type=int, default=1, help='Scaler for channel size')
    parser.add_argument('--num_models', type=int, help='Num of Models if using Ensemble')
    parser.add_argument('--run_num', type=int, help='Run num')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'CNN1D_Att':
        model = CNN1D_Att(num_classes=args.classes, scaler=args.scaler).to(device=device)

    if args.loss_func == 'focal':
        criterion = FocalLoss(gamma=2)
    elif args.loss_func == 'cross_ent':
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    x_data, y_data = load_and_prepare_data(args)

    if args.data == 'ptb_ecg':
        train_portion, test_portion = .7, .2
        # x_data, y_data = load_and_prepare_data(args)
        denoised_x_data = wavelet_denoising(x_data)
        x_corrected_data = median_filtering(denoised_x_data)
        x_normalized = normalization(x_corrected_data)

        X_processed = np.array([signal for signal in tqdm(x_normalized)])
        X_processed = X_processed.reshape(-1, 1, X_processed.shape[1])

    elif args.data == 'mit-bih':
        train_portion, test_portion = .65, .2
        X_processed = x_data.reshape(-1, 1, x_data.shape[1])
    
    y_data = y_data.reshape(-1, 1)

    split_idx1 = int(np.floor(train_portion * X_processed.shape[0]))
    split_idx2 = int(np.floor((1 - train_portion - test_portion) * X_processed.shape[0]))
    split_idx3 = int(np.floor(test_portion * X_processed.shape[0]))

    indices = [i for i in range(X_processed.shape[0])]

    if not args.ensemble:
        np.random.seed(42)
        np.random.shuffle(indices)

        if not os.path.exists(path=f'./{args.data}/non_ensemble/run_{args.run_num}') and not os.path.exists(path=f'./saved_models/{args.model_name}/{args.data}/non_ensemble/run_{args.run_num}'):
            os.mkdir(f'./{args.data}/non_ensemble/run_{args.run_num}') 
            os.mkdir(f'./saved_models/{args.model_name}/{args.data}/non_ensemble/run_{args.run_num}') 

        train_idx, val_idx, test_idx = indices[:split_idx1], indices[split_idx1:split_idx1+split_idx2],\
                                        indices[-split_idx3-1:]
        
        if args.class_balance:
            # SMOTE for class balancing 
            X_train, y_train, X_val, y_val = smote_class_balancing(args, X_processed, y_data, train_idx, val_idx, test_idx)

        elif not args.class_balance:
            X_train, y_train = np.array([X_processed[i,:,:] for i in train_idx]), np.array([y_data[i,:] for i in train_idx])
            X_val, y_val = np.array([X_processed[i,:,:] for i in val_idx]), np.array([y_data[i,:] for i in val_idx])
            X_test, y_test = np.array([X_processed[i,:,:] for i in test_idx]), np.array([y_data[i,:] for i in test_idx])

            pickle.dump((X_test, y_test), open(f'./{args.data}/non_ensemble/run_{args.run_num}/test_idx.pickle', 'wb'))

        print("\Post Class distribution:")
        unique, counts = np.unique(np.vstack((y_train, y_val)), return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(np.vstack((y_train, y_val)))*100:.2f}%)")

        train_dataset = ECGDataset(X_train, y_train)
        val_dataset = ECGDataset(X_val, y_val)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
        run(args, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler)

    elif args.ensemble:
        if not os.path.exists(path=f'./{args.data}/ensemble/run_{args.run_num}') and not os.path.exists(path=f'./saved_models/{args.model_name}/{args.data}/ensemble/run_{args.run_num}'):
            os.mkdir(f'./{args.data}/ensemble/run_{args.run_num}') 
            os.mkdir(f'./saved_models/{args.model_name}/{args.data}/ensemble/run_{args.run_num}') 

        ensembles = {}
        for m in range(args.num_models):
            np.random.seed(42+m)
            np.random.shuffle(indices)

            train_idx, val_idx, test_idx = indices[:split_idx1], indices[split_idx1:split_idx1+split_idx2],\
                                            indices[-split_idx2-1:]
            

            X_train, y_train = np.array([X_processed[i,:,:] for i in train_idx]), np.array([y_data[i,:] for i in train_idx])
            X_val, y_val = np.array([X_processed[i,:,:] for i in val_idx]), np.array([y_data[i,:] for i in val_idx])
            X_test, y_test = np.array([X_processed[i,:,:] for i in test_idx]), np.array([y_data[i,:] for i in test_idx])

            pickle.dump((X_test, y_test), open(f'./{args.data}/ensemble/run_{args.run_num}/test_idx_{m}.pickle', 'wb'))

            train_dataset = ECGDataset(X_train, y_train)
            val_dataset = ECGDataset(X_val, y_val)
            
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
        
            ensemble = run(args, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler)
            ensembles[f'model_{m}'] = ensemble
        torch.save(ensembles, f'./saved_models/{args.model_name}/{args.data}/ensemble/run_{args.run_num}/ecg_ensemble_ptb.pth')
