from math import radians
from pickletools import optimize
import numpy as np
from solver import Solver
from rummikub import Rummikub
import random

class Proteus(object):
    def __init__(self) -> None:
        pass
    
    def get_action(self, state):
        player = state[0,:]
        groups = state[1:,:]
        any_groups_mask = np.any(groups, axis=1)
        any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]

        moves = []
        for group, idx in zip(any_groups, any_groups_idx):
            tiles_idxs = Solver.solve_pair(player, group)
            for tile_idx in tiles_idxs:
                state_test = state.copy()
                state_test[0,tile_idx] = False
                state_test[idx+1, tile_idx] = True
                assessment = self.evaluate_state(state_test)
                moves.append([-1, idx, tile_idx, assessment])

        moves_array = np.array(moves)
        return moves_array[np.argmax(moves_array[:,3]),:]

    def evaluate_state(self, state):
        return random.random()
