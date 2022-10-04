from rummikub import Rummikub
from proteus import Proteus
import numpy as np
import time

if __name__ == '__main__':
    game = Rummikub(2)
    proteus = Proteus(game, './models/first_model')
    state = game.reset()
    while True:
        game.render()
        action, _  = proteus.get_action(state)
        from_group = int(input("From: "))
        to_group = int(input("To: "))
        t_pointer = int(input())
        state_p, _ = game.next_move(from_group, to_group, t_pointer)
        state = state_p
