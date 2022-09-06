from rummikub import Rummikub
from solver import Solver

if __name__ == '__main__':
    game = Rummikub(2)
    while not game.is_end():
        game.render()
        state, reward = game.next_move()
        hint = Solver.solve_pair(state[0:Rummikub.tiles_number], state[Rummikub.tiles_number:Rummikub.tiles_number*2])
        x=0
