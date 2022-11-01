from rummikub import Rummikub
import monte_carlo
import numpy as np
import time
from dataset import DataSet
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-gd", "--gdrive", action="store_true", help="google drive mode")
    args = parser.parse_args()
    if args.gdrive:
        path_prefix = '/content/drive/MyDrive/rummikub/'
    else:
        path_prefix = ''

    game = Rummikub(2, learning=True)
    state = game.reset()
    path_datasets = path_prefix + 'datasets/'
    buffer = DataSet('all', path_datasets)
    positive_buffer = DataSet('positive', path_datasets)
    mc_state = monte_carlo.MonteCarloSearchTreeState(state)
    monte_carlo.MonteCarloTreeSearchNode.create_models(path_prefix)
    while True:
        game.render()
        if game.is_end():
            state = game.reset()
            mc_state = monte_carlo.MonteCarloSearchTreeState(state)
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
