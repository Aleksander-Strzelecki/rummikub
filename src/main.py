from rummikub import Rummikub

if __name__ == '__main__':
    game = Rummikub(2)
    while not game.is_end():
        game.render()
        state, reward = game.next_move()
        x=0
        