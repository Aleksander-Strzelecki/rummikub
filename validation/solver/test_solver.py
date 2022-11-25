from solver import Solver
from rummikub import Rummikub
import numpy as np
import pytest

@pytest.fixture
def groups_test_case():
    groups_test = []
    groups_test_idxs = []
    group_tiles_array = np.zeros((Rummikub.tiles_number,), dtype=bool)
    #0 (1,2,3)
    group_no_jokers = np.copy(group_tiles_array)
    group_no_jokers[[0,1,2]] = True
    groups_test.append(group_no_jokers)
    #1 (2,0,0,5)
    group_two_jokers_inside = group_tiles_array.copy()
    group_two_jokers_inside[[1,4,104,105]] = True
    groups_test.append(group_two_jokers_inside)
    #2 (1,2,0,4)
    group_one_jokers_inside = group_tiles_array.copy()
    group_one_jokers_inside[[0,1,3,104]] = True
    groups_test.append(group_one_jokers_inside)
    #3 (0,0,4,5)
    group_two_jokers_outside = group_tiles_array.copy()
    group_two_jokers_outside[[3,4,104,105]] = True
    groups_test.append(group_two_jokers_outside)
    #4 (0,4,5)
    group_one_jokers_outside = group_tiles_array.copy()
    group_one_jokers_outside[[3,4,105]] = True
    groups_test.append(group_one_jokers_outside)
    #5 (0,4,0,6)
    group_one_jokers_outside_one_inside = group_tiles_array.copy()
    group_one_jokers_outside_one_inside[[3,5,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside)
    #6 (0,0,1,2)
    group_two_jokers_outside_down_limit = group_tiles_array.copy()
    group_two_jokers_outside_down_limit[[0,1,104,105]] = True
    groups_test.append(group_two_jokers_outside_down_limit)
    #7 (0,1,0,3)
    group_one_jokers_outside_one_inside_down_limit = group_tiles_array.copy()
    group_one_jokers_outside_one_inside_down_limit[[0,2,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside_down_limit)
    #8 (0,0,12,13)
    group_two_jokers_outside_up_limit = group_tiles_array.copy()
    group_two_jokers_outside_up_limit[[11,12,104,105]] = True
    groups_test.append(group_two_jokers_outside_up_limit)
    #9 (0,11,0,13)
    group_one_jokers_outside_one_inside_up_limit = group_tiles_array.copy()
    group_one_jokers_outside_one_inside_up_limit[[10,12,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside_up_limit)
    #10 (7,0,9,0,11)
    group_two_jokers_inside_separated = group_tiles_array.copy()
    group_two_jokers_inside_separated[[6,8,10,104,105]] = True
    groups_test.append(group_two_jokers_inside_separated)
    #11 (4,4,4,0)
    group_same_number_joker_full = group_tiles_array.copy()
    group_same_number_joker_full[[3,16,29,104]] = True
    groups_test.append(group_same_number_joker_full)
    #12 (4,4,0)
    group_same_number_joker_three = group_tiles_array.copy()
    group_same_number_joker_three[[3,16,104]] = True
    groups_test.append(group_same_number_joker_three)
    #13 (4,4)
    group_same_number_two = group_tiles_array.copy()
    group_same_number_two[[3,16]] = True
    groups_test.append(group_same_number_two)
    #14 (0)
    group_one_joker = group_tiles_array.copy()
    group_one_joker[[104]] = True
    groups_test.append(group_one_joker)
    #15 ()
    group_empty = group_tiles_array.copy()
    groups_test.append(group_empty)

    for i in range(len(groups_test)):
        groups_test_idxs.append(i)

    return np.array(groups_test)

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
    #8 ()
    group = np.copy(group_tiles_array)
    groups_test.append(group)
    #9 (1b,2b,3b)
    group = np.copy(group_tiles_array)
    group[[13,14,15]] = True
    groups_test.append(group)

    for i in range(len(groups_test)):
        groups_test_idxs.append(i)

    return np.array(groups_test)

# def test_get_rummikub_series_condition(groups_test_case):
#     solver = Solver(groups_test_case)
#     destination_group_tiles_with_idxs = np.array([[3,3,0],
#                                                 [7,8,0],
#                                                 [32,85,104]])
#     source_group_with_idxs = np.array([[3],
#                                     [10],
#                                     [35]])
#     no_joker_tiles_columns = np.array([0,1])                                
#     condition = solver._get_rummikub_series_condition(destination_group_tiles_with_idxs, \
#         source_group_with_idxs, no_joker_tiles_columns, 0)

#     assert condition == np.array([True])


@pytest.mark.parametrize("idx, down, up",[(0, None, None),(1, None, None), (2, None, None), (3, np.array([1,2]), np.array([7,8])),\
    (4, np.array([2]), np.array([7])), (5, np.array([2]), np.array([8])), (6, np.array([]), np.array([4,5])), (7, np.array([]), np.array([5])),\
        (8, np.array([9,10]), np.array([])), (9, np.array([9]), np.array([])), (10, None, None), (11, None, None), (12, None, None), \
            (13, None, None), (14, None, None)])
def test_prepare_groups_cache(groups_test_case, idx, down, up):
    solver = Solver(groups_test_case)
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
    solver = Solver(groups_test_case)
    rummikub_tiles_with_idxs = np.vstack((Rummikub.tiles, np.arange(Rummikub.tiles_number)))
    player_tiles_array = np.ones((Rummikub.tiles_number,), dtype=bool)
    player_tiles_with_idxs = rummikub_tiles_with_idxs[:,player_tiles_array]
    group_tiles_with_idxs = rummikub_tiles_with_idxs[:,groups_test_case[int(request.node.callspec.id)]]
    result = solver.solve_pair(player_tiles_with_idxs, group_tiles_with_idxs, int(request.node.callspec.id))
    assert np.array_equal(result, result_idx) 


def test_solve_manipulation(groups_manipulation_test_case):
    solver = Solver(groups_manipulation_test_case)

    possible_moves = [[0,3,4],[0,3,6],[0,4,4],[0,4,6],\
        [1,0,7],[1,2,7],[1,3,4],[1,4,4],[1,4,7], \
        [2,3,4],[2,3,6],[2,4,4],[2,4,6], \
        [3,0,3],[3,0,104],[3,0,105],[3,1,3],[3,1,104],[3,1,105],[3,2,3],[3,2,104],[3,2,105], \
        [3,4,2],[3,4,3],[3,4,104],[3,4,105],[3,6,104],[3,6,105],[3,9,104],[3,9,105], \
        [4,0,104],[4,1,104],[4,2,104],[4,3,104],[4,6,104],[4,9,104],[5,0,3],[5,1,3],[5,2,3],[5,4,3],[5,4,16],[5,4,29],[5,4,42],[5,9,16], \
        [6,0,104],[6,1,104],[6,2,104],[6,3,5],[6,3,104],[6,4,5],[6,4,18],[6,4,104], \
        [6,9,104],[7,3,0],[7,4,0],[7,4,12],[9,4,13],[9,4,15], \
        [0,8,4],[0,8,6],[1,8,4],[1,8,7],[2,8,4],[2,8,6],[3,8,2],[3,8,3],[3,8,104],[3,8,105],[4,8,104], \
        [5,8,3],[5,8,16],[5,8,29],[5,8,42],[6,8,5],[6,8,18],[6,8,104],[7,8,0],[7,8,12],[9,8,13],[9,8,15]]
    moves = solver.solve_manipulation()
    for move in moves:
        assert move in possible_moves
        possible_moves.remove(move)
    assert possible_moves == []

def test_solve_player_group(groups_test_case):
    solver = Solver(groups_test_case)
    player_tiles_array = np.ones((Rummikub.tiles_number,), dtype=bool)
    possible_moves = [[-1,0,55],[-1,0,104],[-1,0,105],[-1,1,52],[-1,1,57],[-1,1,104],[-1,1,105],[-1,2,56],[-1,2,104],[-1,2,105], \
        [-1,3,52],[-1,3,53],[-1,3,54],[-1,3,57],[-1,3,58],[-1,3,59],[-1,3,104],[-1,3,105],[-1,4,53],[-1,4,54],[-1,4,57],[-1,4,58],[-1,4,104],[-1,4,105], \
        [-1,5,53],[-1,5,54],[-1,5,58],[-1,5,59],[-1,5,104],[-1,5,105],[-1,6,54],[-1,6,55],[-1,6,56],[-1,6,104],[-1,6,105],[-1,7,55], \
        [-1,7,56],[-1,7,104],[-1,7,105],[-1,8,60],[-1,8,61],[-1,8,62],[-1,8,104],[-1,8,105],[-1,9,60],[-1,9,61],[-1,9,104],[-1,9,105], \
        [-1,10,57],[-1,10,63],[-1,10,104],[-1,10,105],[-1,12,94],[-1,12,81],[-1,12,104],[-1,12,105],[-1,13,81],[-1,13,94],[-1,13,104],[-1,13,105]]
    minus_one = np.ones((Rummikub.reduced_tiles_number,1)) * -1
    ones = np.ones((Rummikub.reduced_tiles_number, 1))
    all_tiles = np.arange(Rummikub.reduced_tiles_number-2,Rummikub.tiles_number).reshape((Rummikub.reduced_tiles_number,1))
    possible_moves.extend(np.hstack([minus_one,ones*14,all_tiles]).tolist())
    possible_moves.extend(np.hstack([minus_one,ones*15,all_tiles]).tolist())
    moves = solver.solve_player_groups(player_tiles_array)
    for move in moves:
        assert move in possible_moves
        possible_moves.remove(move)
    assert possible_moves == []
