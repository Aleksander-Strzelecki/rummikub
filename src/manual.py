from rummikub import Rummikub
from proteus import Proteus
import monte_carlo
import numpy as np
import time

if __name__ == '__main__':
    game = Rummikub(2)
    proteus = Proteus(game)
    state = game.reset()
    mc_state = monte_carlo.MonteCarloSearchTreeState(state)
    while True:
        game.render()
        root = monte_carlo.MonteCarloTreeSearchNode(state = mc_state)
        selected_action = root.best_action().parent_action
        print("Best Action: ", selected_action)
        # action, _  = proteus.get_action(state)
        from_group = int(input("From: "))
        to_group = int(input("To: "))
        t_pointer = int(input())
        state_p, _ = game.next_move(from_group, to_group, t_pointer)
        mc_state_p = monte_carlo.MonteCarloSearchTreeState(state_p)
        mc_state = mc_state_p
