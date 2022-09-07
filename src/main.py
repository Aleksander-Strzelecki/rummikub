from rummikub import Rummikub
from proteus import Proteus

if __name__ == '__main__':
    game = Rummikub(2)
    proteus = Proteus()
    while not game.is_end():
        game.render()
        state, reward = game.next_move()
        action = proteus.get_action(state)
        # hint = Solver.solve_pair(state[0:Rummikub.tiles_number], state[Rummikub.tiles_number:Rummikub.tiles_number*2])
        
        x=0
