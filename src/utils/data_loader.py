import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import os

class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def load_data(self):
        if self.dataset == 'cifar10':
            ds, _ = tfds.load(self.dataset, with_info=True, as_supervised=True, split=['train[:80%]', 'train[80%:]', 'test'], shuffle_files=True)
            train_data, val_data, test_data = ds

            x_train, y_train = self.ds_to_numpy(train_data)
            x_val, y_val = self.ds_to_numpy(val_data)
            x_test, y_test = self.ds_to_numpy(test_data)

            return x_train, x_val, x_test, y_train, y_val, y_test
        
        elif self.dataset == 'NusaX':
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(BASE_DIR, 'NusaX')
            train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
            val_df = pd.read_csv(os.path.join(base_path, 'valid.csv'))
            test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

            x_train = train_df.drop(columns=['label']).values
            y_train = train_df['label'].values 
            x_val = val_df.drop(columns=['label']).values
            y_val = val_df['label'].values
            x_test = test_df.drop(columns=['label']).values
            y_test = test_df['label'].values
        
            return np.array(x_train), np.array(x_val), np.array(x_test), np.array(y_train), np.array(y_val), np.array(y_test)

    def ds_to_numpy(self, ds):
        x, y = [], []

        for img, label in ds:
            x.append(np.array(img))
            y.append(np.array(label))

        x = np.array(x)
        y = np.array(y)

        return x, y
    
