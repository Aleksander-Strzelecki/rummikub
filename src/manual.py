from rummikub import Rummikub
import monte_carlo
import numpy as np
import time

if __name__ == '__main__':
    game = Rummikub(2)
    state = game.reset()
    mc_state = monte_carlo.MonteCarloSearchTreeState(state)
    while True:
        game.render()
        root = monte_carlo.MonteCarloTreeSearchNode(state = mc_state)
        actions_sequence = []
        child = root.best_action()
        while(child.children):
            actions_sequence.append(child.parent_action)
            child = child.best_action()
        actions_sequence.append(child.parent_action)
        print("Best Actions: ", actions_sequence)
        # action, _  = proteus.get_action(state)
        from_group = int(input("From: "))
        to_group = int(input("To: "))
        t_pointer = int(input())
        state_p, _ = game.next_move(from_group, to_group, t_pointer)
        actions_sequence.pop(0)
        while actions_sequence:
            game.render()
            print("Best Actions: ", actions_sequence)
            from_group = int(input("From: "))
            to_group = int(input("To: "))
            t_pointer = int(input())
            state_p, _ = game.next_move(from_group, to_group, t_pointer)
            actions_sequence.pop(0)
        mc_state_p = monte_carlo.MonteCarloSearchTreeState(state_p)
        mc_state = mc_state_p
