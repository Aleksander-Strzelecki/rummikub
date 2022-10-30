import numpy as np
from scipy.fftpack import diff

from rummikub import Rummikub

class Solver:
    def __init__(self, groups) -> None:
        any_tile_in_group_mask = np.any(groups, axis=1)
        self._groups_no_empty_idxs = np.where(any_tile_in_group_mask)[0]
        self._groups_no_empty = groups[any_tile_in_group_mask,:]
        if not np.all(any_tile_in_group_mask):
            any_tile_in_group_mask[np.where(any_tile_in_group_mask==False)[0][0]] = True  # add one empty group to evaluation only if place on board
        self._groups_idxs = np.where(any_tile_in_group_mask)[0]
        self._groups = groups[any_tile_in_group_mask,:]
        self._nth_positive_smallest_value_with_joker = {}
        self._nth_valid_largest_value_with_joker = {}
        self._prepare_groups_cache()

    def solve_pair(self, player, group):
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

        rummikub_series_condition = self._get_rummikub_series_condition(group_tiles, player_tiles, no_joker_tile_columns)
        if rummikub_series_condition is not None:
            result = np.hstack([result, player_tiles[2,rummikub_series_condition]])

        rummikub_groups_condition = self._get_rummikub_groups_condition(group_tiles, player_tiles, no_joker_tile_columns)
        if rummikub_groups_condition is not None:
            result = np.hstack([result, player_tiles[2,rummikub_groups_condition]])

        return np.unique(result)

    def solve_manipulation(self):
        moves = []
        groups_tiles_with_idxs = np.vstack((Rummikub.tiles, np.arange(Rummikub.tiles_number)))

        for actual_idx, (group_1, group_1_idx) in enumerate(zip(self._groups, self._groups_idxs)):
            group_1_tiles_with_idxs = groups_tiles_with_idxs[:,group_1]
            no_joker_tile_columns_1 = np.where(group_1_tiles_with_idxs[0,:] > 0)[0]
            if no_joker_tile_columns_1.size == 0:
                group_1_avaliable_tiles = group_1_tiles_with_idxs
            else:
                group_1_avaliable_tiles = self._get_group_avaliable_tiles_manipulation(group_1_tiles_with_idxs, 
                    no_joker_tile_columns_1)

            idx_to_cut = np.where(self._groups_no_empty_idxs == group_1_idx)[0]
            groups_2_slice = np.delete(self._groups_no_empty, idx_to_cut, axis=0)
            groups_2_idxs = np.delete(self._groups_no_empty_idxs, idx_to_cut, axis=0)

            for group_2, group_2_idx in zip(groups_2_slice, groups_2_idxs):
                group_2_tiles_with_idxs = groups_tiles_with_idxs[:,group_2]

                no_joker_tile_columns = np.where(group_2_tiles_with_idxs[0,:] > 0)[0]

                if no_joker_tile_columns.size == 0:
                    for tile_idx in group_1_avaliable_tiles[2,:]:
                        moves.append([group_1_idx, group_2_idx, tile_idx])
                    continue

                rummikub_series_condition = self._get_rummikub_series_condition(group_2_tiles_with_idxs, \
                    group_1_avaliable_tiles, no_joker_tile_columns, group_1_idx)
                if rummikub_series_condition is not None:
                    for tile_idx in group_1_avaliable_tiles[2,rummikub_series_condition]:
                        moves.append([group_1_idx, group_2_idx, tile_idx])

                rummikub_groups_condition = self._get_rummikub_groups_condition(group_2_tiles_with_idxs, \
                    group_1_avaliable_tiles, no_joker_tile_columns)
                if rummikub_groups_condition is not None:
                    for tile_idx in group_1_avaliable_tiles[2,rummikub_groups_condition]:
                        moves.append([group_1_idx, group_2_idx, tile_idx])
        
        return moves

    def solve_no_duplicates(self, player, group):
        tiles_idxs = self.solve_pair(player, group)
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
        jokers_place = diff_array[diff_array > 1]
        jokers_inside = np.sum(jokers_place - 2) + jokers_place.size
        return jokers_inside

    def _get_condition_for_jokers_in(self, jokers_out, group_tiles_with_idxs, no_joker_tile_columns, source_group_tiles_with_idxs, \
        source_group_idx):
        nth_positive_smallest_value_with_joker = self._nth_positive_smallest_value_with_joker.get(source_group_idx)
        nth_valid_largest_value_with_joker = self._nth_valid_largest_value_with_joker.get(source_group_idx)
        condition_jokers_inner_bound = ((source_group_tiles_with_idxs[0,:] == group_tiles_with_idxs[0,no_joker_tile_columns[0]]) \
            & ((np.in1d(source_group_tiles_with_idxs[1,:], nth_positive_smallest_value_with_joker)) | (np.in1d(source_group_tiles_with_idxs[1,:], nth_valid_largest_value_with_joker))))

        return condition_jokers_inner_bound

    def _get_group_avaliable_tiles_manipulation(self, group_tiles_with_idxs, no_joker_tile_columns):
        group_tiles_numbers = group_tiles_with_idxs[1,:]
        jokers_count, jokers_in = self._count_jokers(group_tiles_with_idxs)
        if jokers_in != jokers_count:
            group_avaliable_tiles = group_tiles_with_idxs[:,(group_tiles_numbers == np.amax(group_tiles_numbers)) \
                | (group_tiles_numbers == np.amin(group_tiles_numbers[no_joker_tile_columns])) | (group_tiles_numbers == 0)]
        else:
            group_avaliable_tiles = group_tiles_with_idxs[:,(group_tiles_numbers == np.amax(group_tiles_numbers)) \
                | (group_tiles_numbers == np.amin(group_tiles_numbers[no_joker_tile_columns]))]

        return group_avaliable_tiles

    def _count_jokers(self, group_tiles_with_idxs):
        group_tiles_numbers = group_tiles_with_idxs[1,:]
        jokers_count = len(group_tiles_numbers[group_tiles_numbers == 0])

        if jokers_count > 0:
            jokers_in = self._inner_jokers(group_tiles_numbers)
        else:
            jokers_in = 0

        return jokers_count, jokers_in

    def _get_rummikub_series_condition(self, destination_group_tiles_with_idxs, source_gropup_tiles_with_idxs, no_joker_tile_columns, source_group_idx):
        condition = None

        if np.all((destination_group_tiles_with_idxs[0,:] == \
            destination_group_tiles_with_idxs[0,no_joker_tile_columns[0]]) \
            | (destination_group_tiles_with_idxs[0,:] == 0)) and destination_group_tiles_with_idxs.shape[1] < 13:
            
            condition_jokers_outer_bound = (((source_gropup_tiles_with_idxs[0,:] == destination_group_tiles_with_idxs[0,no_joker_tile_columns[0]]) \
                & ((source_gropup_tiles_with_idxs[1,:] == np.amin(destination_group_tiles_with_idxs[1,no_joker_tile_columns]-1)) \
                | (source_gropup_tiles_with_idxs[1,:] == np.amax(destination_group_tiles_with_idxs[1,no_joker_tile_columns]+1))))\
                | (source_gropup_tiles_with_idxs[0,:] == 0))
            condition = condition_jokers_outer_bound
            jokers_count, jokers_in = self._count_jokers(destination_group_tiles_with_idxs)

            if (jokers_count > 0) and (jokers_count != jokers_in):
                condition_jokers_inner_bound = \
                    self._get_condition_for_jokers_in(jokers_count - jokers_in, \
                        destination_group_tiles_with_idxs, no_joker_tile_columns, source_gropup_tiles_with_idxs, source_group_idx)
                condition = (condition | condition_jokers_inner_bound)
            
        return condition

    def _get_rummikub_groups_condition(self, destination_group_tiles_with_idxs, source_group_tiles_with_idxs, no_joker_tile_columns):
        condition = None

        if np.all((destination_group_tiles_with_idxs[1,:] == destination_group_tiles_with_idxs[1,no_joker_tile_columns[0]]) \
            | (destination_group_tiles_with_idxs[1,:] == 0)) and destination_group_tiles_with_idxs.shape[1] < 4:
            condition = (((source_group_tiles_with_idxs[1,:] == destination_group_tiles_with_idxs[1,no_joker_tile_columns[0]]) \
                & (np.in1d(source_group_tiles_with_idxs[0,:], destination_group_tiles_with_idxs[0,:], invert=True))) \
                    | (source_group_tiles_with_idxs[0,:]==0))

        return condition

    def _get_smallest_largest_value_jokers_in(self, group_tiles_with_idxs, jokers_out):
        no_joker_tile_columns = np.where(group_tiles_with_idxs[0,:] > 0)[0]
        lower_bound_value_with_jokers = group_tiles_with_idxs[1,no_joker_tile_columns]-1*jokers_out-1
        nth_smallest_value_with_joker = np.sort(lower_bound_value_with_jokers)[:jokers_out]
        nth_positive_smallest_value_with_joker = nth_smallest_value_with_joker[nth_smallest_value_with_joker > 0]
        upper_bound_value_with_jokers = group_tiles_with_idxs[1,no_joker_tile_columns]+1*jokers_out+1
        nth_largest_value_with_joker = np.sort(upper_bound_value_with_jokers)[-jokers_out:]
        nth_valid_largest_value_with_joker = nth_largest_value_with_joker[nth_largest_value_with_joker < 14]

        return nth_positive_smallest_value_with_joker, nth_valid_largest_value_with_joker

    def _prepare_groups_cache(self):
        groups_tiles_with_idxs = np.vstack((Rummikub.tiles, np.arange(Rummikub.tiles_number)))

        for group, group_idx in zip(self._groups, self._groups_idxs):
            group_tiles_with_idxs = groups_tiles_with_idxs[:,group]
            no_joker_tile_columns = np.where(group_tiles_with_idxs[0,:] > 0)[0]
            if no_joker_tile_columns.size == 0:
                continue
            if np.all((group_tiles_with_idxs[0,:] == \
            group_tiles_with_idxs[0,no_joker_tile_columns[0]]) \
            | (group_tiles_with_idxs[0,:] == 0)) and group_tiles_with_idxs.shape[1] < 13:
                jokers_count, jokers_in = self._count_jokers(group_tiles_with_idxs)
                jokers_out = jokers_count - jokers_in
                if (jokers_count > 0) and (jokers_count != jokers_in):
                    nth_positive_smallest_value_with_joker, nth_largest_value_with_joker = \
                        self._get_smallest_largest_value_jokers_in(group_tiles_with_idxs, jokers_out)
                    self._nth_valid_largest_value_with_joker[group_idx] = nth_largest_value_with_joker
                    self._nth_positive_smallest_value_with_joker[group_idx] = nth_positive_smallest_value_with_joker
