import numpy as np
import pytest
import monte_carlo
from rummikub import Rummikub

def test_best_action_from_rollout(tmp_path):
    game = Rummikub(2, learning=True, path=tmp_path)
    state = game.reset()
    mc_state = monte_carlo.MonteCarloSearchTreeState(state)
    monte_carlo.MonteCarloTreeSearchNode.create_models(str(tmp_path) + '/')
    root = monte_carlo.MonteCarloTreeSearchNode(state = mc_state)

    mock_rollout_actions1 = [[[0,1,5],[0,1,2],[0,1,3],[0,1,4],[100,0,0]],
                            [[0,1,1],[0,1,2],[0,1,3],[100,0,0]],
                            [[0,2,104],[0,2,1],[0,2,2],[0,2,3],[0,2,4],[0,2,5],[100,0,0]],
                            [[0,1,1],[2,1,2],[100,0,0]]]

    mock_rollout_actions2 = [[[0,1,5],[2,1,2],[3,1,3],[0,1,4],[100,0,0]],
                            [[0,1,1],[0,1,2],[0,1,3],[100,0,0]],
                            [[0,2,104],[1,2,1],[1,2,2],[1,2,3],[1,2,4],[1,2,5],[100,0,0]],
                            [[0,1,1],[2,1,2],[100,0,0]]]

    assert root._best_action_from_rollout(mock_rollout_actions1) == [[0,2,104],[0,2,1],[0,2,2],[0,2,3],[0,2,4],[0,2,5],[100,0,0]]
    assert root._best_action_from_rollout(mock_rollout_actions2) == [[0,1,1],[0,1,2],[0,1,3],[100,0,0]]