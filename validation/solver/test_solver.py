from solver import Solver
from rummikub import Rummikub
import numpy as np
import pytest

@pytest.fixture
def groups_test_case():
    groups_test = []
    groups_test_idxs = []
    group_tiles_array = np.zeros((Rummikub.tiles_number,), dtype=bool)
    #0
    group_no_jokers = np.copy(group_tiles_array)
    group_no_jokers[[0,1,2]] = True
    groups_test.append(group_no_jokers)
    #1
    group_two_jokers_inside = group_tiles_array.copy()
    group_two_jokers_inside[[1,4,104,105]] = True
    groups_test.append(group_two_jokers_inside)
    #2
    group_one_jokers_inside = group_tiles_array.copy()
    group_one_jokers_inside[[0,1,3,104]] = True
    groups_test.append(group_one_jokers_inside)
    #3
    group_two_jokers_outside = group_tiles_array.copy()
    group_two_jokers_outside[[3,4,104,105]] = True
    groups_test.append(group_two_jokers_outside)
    #4
    group_one_jokers_outside = group_tiles_array.copy()
    group_one_jokers_outside[[3,4,105]] = True
    groups_test.append(group_one_jokers_outside)
    #5
    group_one_jokers_outside_one_inside = group_tiles_array.copy()
    group_one_jokers_outside_one_inside[[3,5,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside)
    #6
    group_two_jokers_outside_down_limit = group_tiles_array.copy()
    group_two_jokers_outside_down_limit[[0,1,104,105]] = True
    groups_test.append(group_two_jokers_outside_down_limit)
    #7
    group_one_jokers_outside_one_inside_down_limit = group_tiles_array.copy()
    group_one_jokers_outside_one_inside_down_limit[[0,2,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside_down_limit)
    #8
    group_two_jokers_outside_up_limit = group_tiles_array.copy()
    group_two_jokers_outside_up_limit[[11,12,104,105]] = True
    groups_test.append(group_two_jokers_outside_up_limit)
    #9
    group_one_jokers_outside_one_inside_up_limit = group_tiles_array.copy()
    group_one_jokers_outside_one_inside_up_limit[[10,12,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside_up_limit)
    #10
    group_two_jokers_inside_separated = group_tiles_array.copy()
    group_two_jokers_inside_separated[[6,8,10,104,105]] = True
    groups_test.append(group_two_jokers_inside_separated)
    #11
    group_same_number_joker_full = group_tiles_array.copy()
    group_same_number_joker_full[[3,16,29,104]] = True
    groups_test.append(group_same_number_joker_full)
    #12
    group_same_number_joker_three = group_tiles_array.copy()
    group_same_number_joker_three[[3,16,104]] = True
    groups_test.append(group_same_number_joker_three)
    #13
    group_same_number_two = group_tiles_array.copy()
    group_same_number_two[[3,16]] = True
    groups_test.append(group_same_number_two)
    #14
    group_one_joker = group_tiles_array.copy()
    group_one_joker[[104]] = True
    groups_test.append(group_one_joker)
    #15
    group_empty = group_tiles_array.copy()
    groups_test.append(group_empty)

    for i in range(len(groups_test)):
        groups_test_idxs.append(i)

    return np.array(groups_test), groups_test_idxs

@pytest.fixture
def groups_manipulation_test_case():
    groups_test = []
    groups_test_idxs = []
    group_tiles_array = np.zeros((Rummikub.tiles_number,), dtype=bool)
    #0 (5,6,7)
    group = np.copy(group_tiles_array)
    group[[4,5,6]] = True
    groups_test.append(group)
    #1 (5,0,0,8)
    group = np.copy(group_tiles_array)
    group[[4,7,104,105]] = True
    groups_test.append(group)
    #2 (5,0,7)
    group = np.copy(group_tiles_array)
    group[[4,6,104]] = True
    groups_test.append(group)
    #3 (3,4,0,0)
    group = np.copy(group_tiles_array)
    group[[2,3,104,105]] = True
    groups_test.append(group)
    #4 (0)
    group = np.copy(group_tiles_array)
    group[[104]] = True
    groups_test.append(group)
    #5 (4a,4b,4c,4d)
    group = np.copy(group_tiles_array)
    group[[3,16,29,42]] = True
    groups_test.append(group)
    #6 (6a,6b,0)
    group = np.copy(group_tiles_array)
    group[[5,18,104]] = True
    groups_test.append(group)
    #7 (1,2,3,4,5,6,7,8,9,10,11,12,13)
    group = np.copy(group_tiles_array)
    group[[0,1,2,3,4,5,6,7,8,9,10,11,12]] = True
    groups_test.append(group)
    #8 (1b,2b,3b)
    group = np.copy(group_tiles_array)
    group[[13,14,15]] = True
    groups_test.append(group)
    #9 ()
    group = np.copy(group_tiles_array)
    groups_test.append(group)

    for i in range(len(groups_test)):
        groups_test_idxs.append(i)

    return np.array(groups_test), groups_test_idxs

def test_get_rummikub_series_condition(groups_test_case):
    solver = Solver(groups_test_case[0], groups_test_case[1])
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


@pytest.mark.parametrize("idx, down, up",[(0, None, None),(1, None, None), (2, None, None), (3, np.array([1,2]), np.array([7,8])),\
    (4, np.array([2]), np.array([7])), (5, np.array([2]), np.array([8])), (6, np.array([]), np.array([4,5])), (7, np.array([]), np.array([5])),\
        (8, np.array([9,10]), np.array([])), (9, np.array([9]), np.array([])), (10, None, None), (11, None, None), (12, None, None), \
            (13, None, None), (14, None, None)])
def test_prepare_groups_cache(groups_test_case, idx, down, up):
    solver = Solver(groups_test_case[0], groups_test_case[1])
    if down is not None:
        assert np.array_equal(solver._nth_positive_smallest_value_with_joker.get(idx), down)
    elif down is None:
        assert solver._nth_positive_smallest_value_with_joker.get(idx) is None
    else:
        assert False
    if up is not None:
        assert np.array_equal(solver._nth_valid_largest_value_with_joker.get(idx), up)
    elif up is None:
        assert solver._nth_valid_largest_value_with_joker.get(idx) is None
    else:
        assert False


valid_player_tiles = [np.array([3,55,104,105]), np.array([0,5,52,57,104,105]), np.array([4,56,104,105]),\
    np.array([0,1,2,5,6,7,52,53,54,57,58,59,104,105]), np.array([1,2,5,6,53,54,57,58,104,105]), \
        np.array([1,2,6,7,53,54,58,59,104,105]), np.array([2,3,4,54,55,56,104,105]), \
        np.array([3,4,55,56,104,105]), np.array([8,9,10,60,61,62,104,105]), np.array([8,9,60,61,104,105]), np.array([5,11,57,63,104,105]), \
            np.array([]), np.array([29,42,81,94,104,105]), np.array([29,42,81,94,104,105]), np.array(list(range(106))), np.array(list(range(106)))]
@pytest.mark.parametrize("result_idx", valid_player_tiles, ids=list(range(len(valid_player_tiles))))
def test_solve_pair(groups_test_case, result_idx, request):
    solver = Solver(groups_test_case[0], groups_test_case[1])
    player_tiles_array = np.ones((Rummikub.tiles_number,), dtype=bool)
    result = solver.solve_pair(player_tiles_array, groups_test_case[0][int(request.node.callspec.id)])
    assert np.array_equal(result, result_idx) 


def test_solve_manipulation(groups_manipulation_test_case):
    groups, groups_idxs = groups_manipulation_test_case[0], groups_manipulation_test_case[1]
    groups_no_empty, groups_no_empty_idxs = groups[:-1], groups_idxs[:-1]
    solver = Solver(groups_manipulation_test_case[0], groups_manipulation_test_case[1])

    possible_moves = [[0,1,5],[0,1,6],[0,2,5],[0,3,4],[0,3,5],[0,3,6],[0,4,4],[0,4,5],[0,4,6],\
        [1,0,7],[1,2,7],[1,2,104],[1,2,105],[1,3,4],[1,3,7],[1,4,4],[1,4,7],[2,0,104],[2,1,6], \
        [2,1,104],[2,3,4],[2,3,6],[2,3,104],[2,4,4],[2,4,6],[2,4,104], \
        [2,6,104],[2,8,104],[3,0,3],[3,0,104],[3,0,105],[3,1,3],[3,1,104],[3,1,105],[3,2,3],[3,2,104],[3,2,105], \
        [3,4,2],[3,4,3],[3,4,104],[3,4,105],[3,6,104],[3,6,105],[3,8,104],[3,8,105], \
        [4,0,104],[4,1,104],[4,2,104],[4,3,104],[4,6,104],[4,8,104],[5,0,3],[5,1,3],[5,2,3],[5,4,3],[5,4,16],[5,4,29],[5,4,42],[5,8,16], \
        [6,0,104],[6,1,5],[6,1,104],[6,2,5],[6,2,104],[6,3,5],[6,3,104],[6,4,5],[6,4,18],[6,4,104], \
        [6,5,104],[6,8,104],[7,3,0],[7,4,0],[7,4,12],[8,4,13],[8,4,15], \
        [0,9,4],[0,9,6],[1,9,4],[1,9,7],[2,9,4],[2,9,6],[3,9,2],[3,9,3],[3,9,104],[3,9,105],[4,9,104], \
        [5,9,3],[5,9,16],[5,9,29],[5,9,42],[6,9,5],[6,9,18],[6,9,104],[7,9,0],[7,9,12],[8,9,13],[8,9,15]]
    moves = solver.solve_manipulation(groups_no_empty, groups, groups_no_empty_idxs, groups_idxs)
    for move in moves:
        assert move in possible_moves
