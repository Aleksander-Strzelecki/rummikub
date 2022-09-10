from rummikub import Rummikub
from proteus import Proteus
import numpy as np
import time

if __name__ == '__main__':
    gamma = 0.99
    eps = 0.1
    game = Rummikub(2)
    proteus = Proteus('./models/first_model')
    for episode in range(100):
        state = game.reset()
        buffer = []
        rewards = []
        for i_batch in range(16):
            game.render()
            action = proteus.get_e_greedy_action(state, eps)
            state_p, reward = game.next_move(action[0], action[1], action[2])
            state = state_p
            time.sleep(2)
