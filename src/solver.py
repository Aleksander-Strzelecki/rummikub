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
    def solve_manipulation(cls, groups_from, groups_to, groups_from_idxs, groups_to_idxs):
        moves = []
        groups_tiles_with_idxs = np.vstack((Rummikub.tiles, np.arange(Rummikub.tiles_number)))

        for actual_idx, (group_1, group_1_idx) in enumerate(zip(groups_from, groups_from_idxs)):
            group_1_tiles_with_idxs = groups_tiles_with_idxs[:,group_1]
            group_1_tiles_numbers = group_1_tiles_with_idxs[1,:]
            jokers_count_1 = len(group_1_tiles_numbers[group_1_tiles_numbers == 0])

            if jokers_count_1 > 0:
                jokers_in_1 = cls._inner_jokers(cls, group_1_tiles_numbers)
            else:
                jokers_in_1 = 0

            if jokers_in_1 != jokers_count_1:
                group_1_avaliable_tiles = group_1_tiles_with_idxs[:,(group_1_tiles_numbers == np.amax(group_1_tiles_numbers)) \
                    | (group_1_tiles_numbers == np.amin(group_1_tiles_numbers)) | (group_1_tiles_numbers == 0)]
            else:
                group_1_avaliable_tiles = group_1_tiles_with_idxs[:,(group_1_tiles_numbers == np.amax(group_1_tiles_numbers)) \
                    | (group_1_tiles_numbers == np.amin(group_1_tiles_numbers))]

            groups_2_slice = np.delete(groups_to, actual_idx, axis=0)
            groups_2_idxs = np.delete(groups_to_idxs, actual_idx, axis=0)

            for group_2, group_2_idx in zip(groups_2_slice, groups_2_idxs):
                group_2_tiles_with_idxs = groups_tiles_with_idxs[:,group_2]
                group_2_tiles_numbers = group_2_tiles_with_idxs[1,:]

                no_joker_tile_columns = np.where(group_2_tiles_with_idxs[0,:] > 0)[0]

                if no_joker_tile_columns.size == 0:
                    for tile_idx in group_1_avaliable_tiles[2,:]:
                        moves.append([group_1_idx, group_2_idx, tile_idx])
                    continue

                no_joker_tile_column = no_joker_tile_columns[0]

                if np.all((group_2_tiles_with_idxs[0,:] == group_2_tiles_with_idxs[0,no_joker_tile_column]) \
                    | (group_2_tiles_with_idxs[0,:] == 0)) and group_2_tiles_with_idxs.shape[1] < 13:
                    condition_jokers_outer_bound = (((group_1_avaliable_tiles[0,:] == group_2_tiles_with_idxs[0,no_joker_tile_column]) \
                        & ((group_1_avaliable_tiles[1,:] == np.amin(group_2_tiles_with_idxs[1,no_joker_tile_columns]-1)) | \
                            (group_1_avaliable_tiles[1,:] == np.amax(group_2_tiles_with_idxs[1,no_joker_tile_columns]+1))))\
                        | (group_1_avaliable_tiles[0,:] == 0))
                    condition = condition_jokers_outer_bound
                    jokers_count_2 = len(group_2_tiles_numbers[group_2_tiles_numbers == 0])

                    if jokers_count_2 > 0:
                        jokers_in_2 = cls._inner_jokers(cls, group_2_tiles_numbers)
                    else:
                        jokers_in_2 = jokers_count_2

                    if (jokers_count_2 > 0) and (jokers_count_2 != jokers_in_2):
                        jokers_out_2 = jokers_count_2 - jokers_in_2
                        lower_bound_value_with_jokers_2 = group_2_tiles_with_idxs[1,no_joker_tile_columns]-1*jokers_out_2-1
                        nth_smallest_value_with_joker_2 = np.sort(lower_bound_value_with_jokers_2)[:jokers_out_2]
                        nth_positive_smallest_value_with_joker_2 = nth_smallest_value_with_joker_2[nth_smallest_value_with_joker_2 > 0]
                        upper_bound_value_with_jokers_2 = group_2_tiles_with_idxs[1,no_joker_tile_columns]+1*jokers_out_2+1
                        nth_largest_value_with_joker_2 = np.sort(upper_bound_value_with_jokers_2)[-jokers_out_2:]
                        nth_valid_largest_value_with_joker_2 = nth_largest_value_with_joker_2[nth_largest_value_with_joker_2 < 14]
                        condition_jokers_inner_bound = ((group_1_avaliable_tiles[0,:] == group_2_tiles_with_idxs[0,no_joker_tile_column]) \
                            & ((np.in1d(group_1_avaliable_tiles[1,:], nth_positive_smallest_value_with_joker_2)) | (np.in1d(group_1_avaliable_tiles[1,:], nth_valid_largest_value_with_joker_2))))
                        condition = (condition | condition_jokers_inner_bound)

                    for tile_idx in group_1_avaliable_tiles[2,condition]:
                        moves.append([group_1_idx, group_2_idx, tile_idx])

                if np.all((group_2_tiles_with_idxs[1,:] == group_2_tiles_with_idxs[1,no_joker_tile_column]) | (group_2_tiles_with_idxs[1,:] == 0)) and group_2_tiles_with_idxs.shape[1] < 4:
                    condition = (((group_1_avaliable_tiles[1,:] == group_2_tiles_with_idxs[1,no_joker_tile_column]) & (np.in1d(group_1_avaliable_tiles[0,:], group_2_tiles_with_idxs[0,:], invert=True))) \
                        | (group_1_avaliable_tiles[0,:]==0))

                    for tile_idx in group_1_avaliable_tiles[2,condition]:
                        moves.append([group_1_idx, group_2_idx, tile_idx])
        
        return moves

    @classmethod
    def solve_no_duplicates(cls, player, group):
        tiles_idxs = cls.solve_pair(player, group)
        tiles_idxs_bool = np.zeros((Rummikub.tiles_number), dtype=bool)
        tiles_idxs_bool[tiles_idxs] = True
        tiles_idxs_bool[:Rummikub.reduced_tiles_number-2] = tiles_idxs_bool[:Rummikub.reduced_tiles_number-2] \
            & np.logical_xor(tiles_idxs_bool[:Rummikub.reduced_tiles_number-2], tiles_idxs_bool[Rummikub.reduced_tiles_number-2:Rummikub.tiles_number-2])

        return np.where(tiles_idxs_bool)[0]

    @classmethod
    def check_board(cls, groups):
        count = np.sum(groups, axis=1)
        count_no_zero = np.where(count==0, 3, count)
        return np.all(count_no_zero > 2)

    def _inner_jokers(self, array):
        array_no_joker = array[array != 0]
        diff_array = np.diff(np.sort(array_no_joker))
        return np.sum(diff_array // 2)
