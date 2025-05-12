import torch
import os
import argparse
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import ECGDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix
from models.ecg_model import CNN1D_Att


def non_ensemble_eval(args, model, test_dataloader):
    model.eval()
    all_y, all_y_hat = [], []
    for batch in test_dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            # Add batch to GPU
            outputs = model(inputs)
            y = labels.detach().cpu().numpy().tolist()
            y_pred = outputs.detach().cpu().numpy().tolist()
            all_y.extend(y)
            all_y_hat.extend(y_pred)

    average_type = 'binary' if args.classes == 2 else 'micro' 
    target_names = ['Normal', 'Abnormal'] if args.data == 'ptb_ecg' else ['Normal', 'S', 'V', 'F', 'Q']

    final_all_y_pred = np.argmax(all_y_hat, axis=1)
    print(f'Test Accuracy Score is: {round(accuracy_score(all_y, final_all_y_pred), 4)}')
    print(f'Test Precision Score is: {round(precision_score(all_y, final_all_y_pred, average=average_type), 4)}')
    print(f'Test Recall Score is: {round(recall_score(all_y, final_all_y_pred, average=average_type), 4)}')
    print(f'Test F1 Score is: {round(f1_score(all_y, final_all_y_pred, average=average_type), 4)}')
    print('----------------Classification Report ---------------------')
    print(classification_report(all_y, final_all_y_pred, target_names=target_names))

    cm = confusion_matrix(all_y, final_all_y_pred)
    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    plt.figure()
    sns.heatmap(cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax, fmt="d")
    sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]), fmt="d")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.show()


def ensemble_eval(model, ensemble, test_dataloader, y_test):
    model_accuracies = []
    model_predictions = []
    with torch.no_grad():
        for i, model_dict in enumerate(ensemble):            
            y_preds = []
            y_trues = []
            model.load_state_dict(ensemble['model_{}'.format(i)][0]['model_state_dict'])
            model.eval()

            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                y_preds.extend(outputs.detach().cpu().numpy().tolist())
                y_trues.extend(labels.detach().cpu().numpy().tolist())

            final_all_y_pred = np.argmax(y_preds, axis=1)
            accuracy = round(accuracy_score(y_trues, final_all_y_pred), 4)
            model_predictions.append(y_preds)
            model_accuracies.append(accuracy)
    
    # Weighted average of predictions
    weights = np.array(model_accuracies)
    weights = weights / weights.sum()
   
    ensemble_predictions = np.average(model_predictions, weights=weights, axis=0)
    ensemble_classes = np.argmax(ensemble_predictions, axis=1)
    
    cm = confusion_matrix(y_test, ensemble_classes)
    precision = np.diag(cm) / np.sum(cm, axis=0) 
    recall = np.diag(cm) / np.sum(cm, axis=1)    

    accuracy = np.trace(cm)/np.sum(cm)
    f1_per_class = 2 * (precision * recall) / (precision + recall)

    return accuracy, f1_per_class
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_name', type=str, help='model')
    parser.add_argument('--data', type=str, help='dataset')
    parser.add_argument('--ensemble', type=bool, default=False, help='algorithm')
    parser.add_argument('--classes', type=int, help='Num of classes')
    parser.add_argument('--scaler', type=int, help='scaler for model size')
    parser.add_argument('--num_models', type=int, help='Num of Models if using Ensemble')
    parser.add_argument('--run_num', type=int, help='Run num')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'CNN1D_Att':
        model = CNN1D_Att(num_classes=args.classes, scaler=args.scaler).to(device)

    if (args.data == 'ptb_ecg' or args.data == 'mit-bih') and not args.ensemble:
        X_test, y_test = pickle.load(open(f'./{args.data}/non_ensemble/run_{args.run_num}/test_idx.pickle', 'rb'))
        test_dataset = ECGDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        model.load_state_dict(torch.load(f'./saved_models/{args.model_name}/{args.data}/non_ensemble/run_{args.run_num}/best_model_num_{args.run_num}.pt', map_location=device)['model_state_dict'])
        non_ensemble_eval(args, model, test_dataloader)
    

    elif args.data == 'ptb_ecg' and args.ensemble:
        ensemble = torch.load(f'./saved_models/{args.model_name}/ensemble/run_{args.run_num}/ecg_ensemble_ptb.pth')
        test_data_files = os.listdir(f'./{args.data}/ensemble/run_{args.run_num}')
        all_acc, all_f1 = [], []
        for file in test_data_files:
            X_test, y_test = pickle.load(open(f'./{args.data}/ensemble/run_{args.run_num}/{file}', 'rb'))
            test_dataset = ECGDataset(X_test, y_test)
            test_dataloader = DataLoader(test_dataset, batch_size=32)
            acc, f1 = ensemble_eval(model, ensemble, test_dataloader, y_test)
            all_acc.append(acc)
            all_f1.append(f1)

        print(f'Ensemble Accuracy and F1 Scores are: {round(np.mean(np.array(all_acc)), 3)} and {round(np.mean(np.array(all_f1)), 3)}')

    


# python eval.py --batch_size 32 --model_name CNN1D_Att --data mit-bih --classes 5 --scaler 1 --num_models 5 --run_num 1
