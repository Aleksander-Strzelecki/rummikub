import numpy as np

class DataSet():
    def __init__(self):
        self.x = []
        self.y = []

    def extend_dataset(self, x, y):
        self.x.extend(x)
        self.y.extend(y)

    def get_data(self):
        return np.array(self.x), np.array(self.y)
