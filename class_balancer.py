from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import pickle


def smote_class_balancing(args, X_data, y_data, train_idx, val_idx, test_idx):

    data = np.hstack((np.squeeze(X_data, axis=1), y_data))
    data = pd.DataFrame(data)
    train_data, test_data = data.loc[train_idx + val_idx], data.loc[test_idx]
    train_data_0 = train_data.loc[(train_data[300] == 0)]
    train_data_1234 = train_data.loc[(train_data[300] != 0)]

    train_data_0_resampled=train_data_0.sample(n=50000,random_state=42)

    # convert dataframe to numpy array
    train_data_0_resampled = train_data_0_resampled.to_numpy()

    # 2. Class 1, 2, 3, 4: Use SMOTE to oversample upto 50000 data
    # converting from df to np ndarray
    train_data_1234_arr = train_data_1234.to_numpy()
    X_4cl, y_4cl = train_data_1234_arr[:, :-1], train_data_1234_arr[:, -1]


    strategy = {1:50000, 2:50000, 3:50000, 4:50000}
    oversample = SMOTE(sampling_strategy=strategy)
    sub_X, sub_y = oversample.fit_resample(X_4cl, y_4cl)

    sub_y = sub_y.reshape(-1, 1)
    train_data_1234_resampled = np.hstack((sub_X, sub_y))

    train_data_resampled = np.vstack((train_data_0_resampled, train_data_1234_resampled))

    new_indices = [i for i in range(len(train_data_resampled))]

    np.random.seed(42)
    np.random.shuffle(new_indices)

    split_idx1 = int(np.floor(.65 * train_data_resampled.shape[0]))
    split_idx2 = int(np.floor(.15 * train_data_resampled.shape[0]))
    new_train_idx, new_val_idx = new_indices[:split_idx1], new_indices[split_idx1:split_idx1+split_idx2]
    new_X_data, new_y_data = np.expand_dims(train_data_resampled[:,:-1], axis=1), np.expand_dims(train_data_resampled[:,-1], axis=1)

    X_train, y_train = np.array([new_X_data[i,:,:] for i in new_train_idx]), np.array([new_y_data[i,:] for i in new_train_idx])
    X_val, y_val = np.array([new_X_data[i,:,:] for i in new_val_idx]), np.array([new_y_data[i,:] for i in new_val_idx])

    X_test, y_test = np.expand_dims(test_data.values[:,:-1], axis=1), np.expand_dims(test_data.values[:,-1], axis=1)
    
    pickle.dump((X_test, y_test), open(f'./{args.data}/non_ensemble/run_{args.run_num}/test_idx_balanced.pickle', 'wb'))

    return X_train, y_train, X_val, y_val