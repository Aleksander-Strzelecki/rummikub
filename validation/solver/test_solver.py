from solver import Solver
import numpy as np

solver = Solver()
destination_group_tiles_with_idxs = np.array([[3,3,0],
                                              [7,8,0],
                                              [32,85,104]])
source_group_with_idxs = np.array([[3],
                                   [10],
                                   [35]])
no_joker_tiles_columns = np.array([0,1])                                
condition = solver._get_rummikub_series_condition(destination_group_tiles_with_idxs, \
    source_group_with_idxs, no_joker_tiles_columns)

assert condition == np.array([True])