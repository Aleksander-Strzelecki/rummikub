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

    def shrink(self, size):
        x_np = np.array(self.x)
        y_np = np.array(self.y)
        x_np = x_np[-size:]
        y_np = y_np[-size:]
        self.x = x_np.tolist()
        self.y = y_np.tolist()

    def length(self):
        return len(self.x)
