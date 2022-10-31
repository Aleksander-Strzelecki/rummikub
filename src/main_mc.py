from rummikub import Rummikub
import monte_carlo
import numpy as np
import time
from dataset import DataSet

if __name__ == '__main__':
    game = Rummikub(2, learning=True)
    state = game.reset()
    buffer = DataSet()
    positive_buffer = DataSet()
    mc_state = monte_carlo.MonteCarloSearchTreeState(state)
    monte_carlo.MonteCarloTreeSearchNode.create_models()
    while True:
        game.render()
        root = monte_carlo.MonteCarloTreeSearchNode(state = mc_state)
        actions_sequence, buffer = root.best_actions(buffer=buffer, positive_buffer=positive_buffer)
        print("Best Actions: ", actions_sequence)
        actions_sequence = np.array(actions_sequence)
        actions_sequence[np.where(actions_sequence[:,0] < 100)[0], 0:2] -= 1
        actions_sequence = actions_sequence.tolist()
        from_group, to_group, t_pointer = actions_sequence.pop(0)
        state_p, _ = game.next_move(from_group, to_group, t_pointer)
        while actions_sequence:
            # game.render()
            # time.sleep(2)
            from_group, to_group, t_pointer = actions_sequence.pop(0)
            state_p, _ = game.next_move(from_group, to_group, t_pointer)
        mc_state_p = monte_carlo.MonteCarloSearchTreeState(state_p)
        mc_state = mc_state_p
