from rummikub import Rummikub
import monte_carlo
import numpy as np
import time
from dataset import DataSet
import argparse
import global_variables.tensorboard_variables as tbv
from google.colab import drive

def update_tensorboard_player_tiles_counter(game:Rummikub):
    for i in range(game.num_players):
        ith_player_tiles_count = game.get_player_tiles_number(i)
        tbv.tensorboard_player_tiles_counter[i] = ith_player_tiles_count

def count_manipulations(actions_sequence:np.ndarray):
    from_array = actions_sequence[:,0]
    manipulation_bool = ((from_array > 0) & (from_array < 100))
    
    return manipulation_bool.sum()

def update_tensorboard_manipulation_counter(game:Rummikub, actions_sequence:np.ndarray):
    activ_player_idx = game.activ
    activ_player_manipulation = count_manipulations(actions_sequence)
    tbv.tensorboard_manipulation_counter_player[activ_player_idx] = activ_player_manipulation
    tbv.tensorboard_manipulation_counter = activ_player_manipulation

def update_google_drive():
    drive.flush_and_unmount()
    drive.mount('/content/drive')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-gd", "--gdrive", action="store_true", help="google drive mode")
    args = parser.parse_args()
    if args.gdrive:
        path_prefix = '/content/drive/MyDrive/rummikub/'
        drive.mount('/content/drive')
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
        update_tensorboard_player_tiles_counter(game)
        update_tensorboard_manipulation_counter(game, actions_sequence)
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
        if args.gdrive:
            update_google_drive()
