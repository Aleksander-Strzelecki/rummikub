from rummikub import Rummikub
from proteus import Proteus
import time

if __name__ == '__main__':
    game = Rummikub(2)
    proteus = Proteus()
    state = game.reset()
    for t in range(200):
        game.render()
        action = proteus.get_e_greedy_action(state)
        state, reward = game.next_move(action[0], action[1], action[2])
        time.sleep(5)
        # hint = Solver.solve_pair(state[0:Rummikub.tiles_number], state[Rummikub.tiles_number:Rummikub.tiles_number*2])
        
        x=0
