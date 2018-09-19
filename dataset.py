import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, path):
        df = pd.read_csv(path)
        x, y = np.split(df, [len(df.columns)-1], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.5)
        self.x_validation, self.x_test, self.y_validation, self.y_test = train_test_split(self.x_test, self.y_test, test_size=0.5)
        
    def get_train_set(self):
        return self.x_train, self.y_train

    def get_validation_set(self):
        return self.x_validation, self.y_validation

    def get_test_set(self):
        return self.x_test, self.y_test