from solver import Solver
from rummikub import Rummikub
import numpy as np
import pytest

@pytest.fixture
def groups_test_case():
    groups_test = []
    groups_test_idxs = []
    group_tiles_array = np.zeros((Rummikub.tiles_number,), dtype=bool)
    
    group_no_jokers = np.copy(group_tiles_array)
    group_no_jokers[[0,1,2]] = True
    groups_test.append(group_no_jokers)

    group_two_jokers_inside = group_tiles_array.copy()
    group_two_jokers_inside[[1,4,104,105]] = True
    groups_test.append(group_two_jokers_inside)

    group_one_jokers_inside = group_tiles_array.copy()
    group_one_jokers_inside[[0,1,3,104]] = True
    groups_test.append(group_one_jokers_inside)

    group_two_jokers_outside = group_tiles_array.copy()
    group_two_jokers_outside[[3,4,104,105]] = True
    groups_test.append(group_two_jokers_outside)

    group_one_jokers_outside = group_tiles_array.copy()
    group_one_jokers_outside[[3,4,105]] = True
    groups_test.append(group_one_jokers_outside)

    group_one_jokers_outside_one_inside = group_tiles_array.copy()
    group_one_jokers_outside_one_inside[[3,5,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside)

    group_two_jokers_outside_down_limit = group_tiles_array.copy()
    group_two_jokers_outside_down_limit[[0,1,104,105]] = True
    groups_test.append(group_two_jokers_outside_down_limit)

    group_one_jokers_outside_one_inside_down_limit = group_tiles_array.copy()
    group_one_jokers_outside_one_inside_down_limit[[0,2,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside_down_limit)

    group_two_jokers_outside_up_limit = group_tiles_array.copy()
    group_two_jokers_outside_up_limit[[11,12,104,105]] = True
    groups_test.append(group_two_jokers_outside_up_limit)

    group_one_jokers_outside_one_inside_up_limit = group_tiles_array.copy()
    group_one_jokers_outside_one_inside_up_limit[[10,12,104,105]] = True
    groups_test.append(group_one_jokers_outside_one_inside_up_limit)

    group_two_jokers_inside_separated = group_tiles_array.copy()
    group_two_jokers_inside_separated[[6,8,10,104,105]] = True
    groups_test.append(group_two_jokers_inside_separated)

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
        (8, np.array([9,10]), np.array([])), (9, np.array([9]), np.array([])), (10, None, None)])
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
        np.array([3,4,55,56,104,105]), np.array([8,9,10,60,61,62,104,105]), np.array([8,9,60,61,104,105]), np.array([5,11,57,63,104,105])]
@pytest.mark.parametrize("result_idx", valid_player_tiles, ids=list(range(len(valid_player_tiles))))
def test_solve_pair(groups_test_case, result_idx, request):
    solver = Solver(groups_test_case[0], groups_test_case[1])
    player_tiles_array = np.ones((Rummikub.tiles_number,), dtype=bool)
    result = solver.solve_pair(player_tiles_array, groups_test_case[0][int(request.node.callspec.id)])
    assert np.array_equal(result, result_idx) 
