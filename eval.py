import torch
import argparse
import pickle
import numpy as np
from utils import ECGDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from models.ecg_model import CNN1D_Att


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_name', type=str, help='model')
    parser.add_argument('--data', type=str, help='dataset')
    parser.add_argument('--ensemble', type=bool, help='algorithm')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--classes', type=int, help='Num of classes')
    parser.add_argument('--run_num', type=int, help='Run num')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'CNN1D_Att':
        model = CNN1D_Att(num_classes=args.classes).to(device)

    if args.data == 'ptb_ecg':
        X_test, y_test = pickle.load(open(f'./{args.data}/non_ensemble/run_{args.run_num}/test_idx.pickle', 'rb'))
        test_dataset = ECGDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=32)

    model.load_state_dict(torch.load('./saved_models/{}/best_model_num_{}.pt'.format(args.model_name, args.run_num), map_location=device)['model_state_dict'])

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


    final_all_y_pred = np.argmax(all_y_hat, axis=1)
    print(f'Test Accuracy Score is: {round(accuracy_score(all_y, final_all_y_pred), 4)}')
    print(f'Test Precision Score is: {round(precision_score(all_y, final_all_y_pred), 4)}')
    print(f'Test Recall Score is: {round(recall_score(all_y, final_all_y_pred), 4)}')
    print(f'Test F1 Score is: {round(f1_score(all_y, final_all_y_pred), 4)}')
    print('----------------Classification Report ---------------------')
    print(classification_report(all_y, final_all_y_pred, target_names=['Normal', 'Abnormal']))

# python eval.py --batch_size 32 --model_name CNN1D_Att --data ptb_ecg --ensemble False --classes 2 --run_num 1
