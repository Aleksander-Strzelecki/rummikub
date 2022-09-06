import numpy as np

from rummikub import Rummikub

class Solver:
    def __init__(self) -> None:
        pass

    @classmethod
    def solve_pair(cls, player, group):
        player_idx = np.nonzero(player)[0]
        group_idx = np.nonzero(group)[0]
        if group_idx.size == 0:
            return player_idx
        # 0 row colors
        # 1 row values
        # 2 row tile id
        player_tiles = np.vstack([Rummikub.tiles[:, player_idx], player_idx])
        group_tiles = np.vstack([Rummikub.tiles[:, group_idx], group_idx])
        result = np.array([], dtype=int)

        if np.all(group_tiles[0,:] == group_tiles[0,0]) and group_tiles.shape[1] < 13:
            condition = (((player_tiles[0,:] == group_tiles[0,0]) & ((player_tiles[1,:] == np.amin(group_tiles[1,:]-1)) | (player_tiles[1,:] == np.amax(group_tiles[1,:]+1))))\
                          | (player_tiles[0,:] == 0))
            result = np.hstack([result, player_tiles[2,condition]])
        if np.all(group_tiles[1,:] == group_tiles[1,0]) and group_tiles.shape[1] < 4:
            condition = (((player_tiles[1,:] == group_tiles[1,0]) & (np.in1d(player_tiles[0,:], group_tiles[0,:], invert=True))) | (player_tiles[0,:]==0))
            result = np.hstack([result, player_tiles[2,condition]])
        return np.unique(result)
