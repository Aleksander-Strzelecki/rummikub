from rummikub import Rummikub
from proteus import Proteus
import numpy as np
import time

if __name__ == '__main__':
    gamma = 0.90
    eps = 0.2
    game = Rummikub(2)
    proteus = Proteus(game)
    for episode in range(501):
        state = game.reset()
        for length in range(10):
            buffer = []
            rewards = []
            for i_batch in range(64):
                # game.render()
                action, max_s = proteus.get_e_greedy_action(state, eps)
                state_p, reward = game.next_move(action[0], action[1], action[2])
                rewards.append(reward)
                buffer.append((state, max_s, reward))
                state = state_p
                # time.sleep(5)
                # hint = Solver.solve_pair(state[0:Rummikub.tiles_number], state[Rummikub.tiles_number:Rummikub.tiles_number*2])
                # x=0
            print('Sum of rewards {}'.format(sum(rewards)))
            # dataset = []
            x_train = []
            y_train = []
            for exp in buffer:
                s, max_s, r = exp
                target = r + gamma * max_s
                x_train.append(s)
                y_train.append(target)
                # dataset.append([s, target])
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            proteus.update_batch((x_train, y_train))
            # if episode % 50 == 0:
        proteus.save_model('./models/first_model')
