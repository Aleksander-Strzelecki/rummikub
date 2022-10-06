import numpy as np

from rummikub import Rummikub

class Solver:
    def __init__(self) -> None:
        pass

    @classmethod
    def solve_pair(cls, player, group):
        player_tiles_idx = np.nonzero(player)[0]
        group_tiles_idx = np.nonzero(group)[0]
        if group_tiles_idx.size == 0:
            return player_tiles_idx
        # 0 row colors
        # 1 row values
        # 2 row tile id
        player_tiles = np.vstack([Rummikub.tiles[:, player_tiles_idx], player_tiles_idx])
        group_tiles = np.vstack([Rummikub.tiles[:, group_tiles_idx], group_tiles_idx])
        result = np.array([], dtype=int)

        no_joker_tile_columns = np.where(group_tiles[0,:] > 0)[0]
        if no_joker_tile_columns.size == 0:
            return player_tiles_idx
        no_joker_tile_column = no_joker_tile_columns[0]

        if np.all((group_tiles[0,:] == group_tiles[0,no_joker_tile_column]) | (group_tiles[0,:] == 0)) and group_tiles.shape[1] < 13:
            group_numbers = group_tiles[1,:]
            condition_jokers_outer_bound = (((player_tiles[0,:] == group_tiles[0,no_joker_tile_column]) & ((player_tiles[1,:] == np.amin(group_tiles[1,no_joker_tile_columns]-1)) | (player_tiles[1,:] == np.amax(group_tiles[1,no_joker_tile_columns]+1))))\
                        | (player_tiles[0,:] == 0))
            condition = condition_jokers_outer_bound
            jokers_count = len(group_numbers[group_numbers == 0])
            jokers_in = cls._inner_jokers(cls, group_numbers)
            if (jokers_count > 0) and (jokers_count != jokers_in):
                jokers_out = jokers_count - jokers_in
                lower_bound_value_with_jokers = group_tiles[1,no_joker_tile_columns]-1*jokers_out-1
                nth_smallest_value_with_joker = np.sort(lower_bound_value_with_jokers)[:jokers_out]
                nth_positive_smallest_value_with_joker = nth_smallest_value_with_joker[nth_smallest_value_with_joker > 0]
                upper_bound_value_with_jokers = group_tiles[1,no_joker_tile_columns]+1*jokers_out+1
                nth_largest_value_with_joker = np.sort(upper_bound_value_with_jokers)[-jokers_out:]
                nth_valid_largest_value_with_joker = nth_largest_value_with_joker[nth_largest_value_with_joker < 14]
                condition_jokers_inner_bound = ((player_tiles[0,:] == group_tiles[0,no_joker_tile_column]) & ((np.in1d(player_tiles[1,:], nth_positive_smallest_value_with_joker)) | (np.in1d(player_tiles[1,:], nth_valid_largest_value_with_joker))))
                condition = (condition | condition_jokers_inner_bound)
            result = np.hstack([result, player_tiles[2,condition]])
        if np.all((group_tiles[1,:] == group_tiles[1,no_joker_tile_column]) | (group_tiles[1,:] == 0)) and group_tiles.shape[1] < 4:
            condition = (((player_tiles[1,:] == group_tiles[1,no_joker_tile_column]) & (np.in1d(player_tiles[0,:], group_tiles[0,:], invert=True))) | (player_tiles[0,:]==0))
            result = np.hstack([result, player_tiles[2,condition]])
        return np.unique(result)

    @classmethod
    def check_board(cls, groups):
        count = np.sum(groups, axis=1)
        count_no_zero = np.where(count==0, 3, count)
        return np.all(count_no_zero > 2)

    def _inner_jokers(self, array):
        array_no_joker = array[array != 0]
        diff_array = np.diff(np.sort(array_no_joker))
        return np.sum(diff_array // 2)
