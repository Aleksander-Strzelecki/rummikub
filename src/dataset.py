import numpy as np
import json
import os
import global_variables.tensorboard_variables as tbv

class DataSet():
    def __init__(self, name, path):
        self.x = []
        self.y = []
        self.name = name
        self._path = path
        self.load()

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

    def save(self):
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        with open(self._path + 'dataset_' + self.name, "w+") as fp:
            json.dump(d, fp)

    def load(self):
        path_to_file = self._path + 'dataset_' + self.name
        if os.path.exists(path_to_file):
            print('loading {} dataset from file {}'.format(self.name, path_to_file))
            with open(path_to_file, "r") as fp:
                d = json.load(fp)
            self.x = d['x']
            self.y = d['y']
        else:
            print('creating new {} dataset'.format(self.name))
            os.mkdir(self._path)

    def tensorboard_update(self):
        tbv.tensorboard_buffer_elements[self.name] = len(self.x)
