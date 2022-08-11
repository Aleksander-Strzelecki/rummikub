from termcolor import colored
import numpy as np

class Rummikub:
    def __init__(self, num_players) -> None:
        self.activ = 0
        self.num_players = num_players
        self.players = []
        for i in range(num_players):
            self.players.append(Player())
        self.tiles_pointers = np.ones((106), dtype=bool)
        self.tiles = {}
        self.create_tiles()
        self.distribute_tiles()
        self.groups = []
        self.create_groups()
        
    def create_tiles(self):
        colors = ['red', 'red', 'white','white', 'yellow', 'yellow','blue', 'blue']
        counter = 0
        for color in colors:
            for i in range(1,14):
                self.tiles[counter] = (color, i)
                counter += 1
        self.tiles[counter] = ('magenta', 0)
        counter += 1
        self.tiles[counter] = ('magenta', 0)

    def distribute_tiles(self):
        indexes = np.random.choice(106, 14*len(self.players), replace=False)
        # print(len(indexes))
        for idx, player in enumerate(self.players):
            choice = indexes[14*idx:14*(idx+1)]
            # print(choice, len(choice))
            player.set_tiles(choice)
            self.tiles_pointers[choice] = False

    def render(self):
        # print(len(np.nonzero(self.tiles_pointers)[0]))
        # for player in self.players:
        #     print(len(np.nonzero(player.get_pointers())[0]))
        free_idx = np.nonzero(self.tiles_pointers)[0]
        print("Free tiles[{}]: ".format(len(free_idx)))
        for idx in free_idx:
            tile = self.tiles[idx]
            print(colored(tile[1], tile[0]), end=" ")
        print('')
        for number, player in enumerate(self.players):
            player_idx = np.nonzero(player.get_pointers())[0]
            print(f"Player {number}[{len(player_idx)}]: ")
            for idx in player_idx:
                tile = self.tiles[idx]
                print(colored(tile[1], tile[0]), end=" ")
            print('')

    def create_groups(self):
        for i in range(36):
            self.groups.append(np.zeros((106), dtype=bool))

    def next_move(self):
        actual_player = self.players[self.activ]
        from_group = int(input("From: "))
        to_group = int(input("To: "))
        pointers = []
        n = int(input("Enter number of elements : "))
        for i in range(0, n):
            ele = int(input())
            pointers.append(ele) 
        self.groups[from_group][pointers] = False
        self.groups[to_group][pointers] = True
        self.activ = (self.activ + 1) % self.num_players

    def is_end(self):
        return False


class Handler(object):
    def __init__(self) -> None:
        self.tiles_pointers = np.zeros((106), dtype=bool)

    def set_tiles(self, idx):
        self.tiles_pointers[idx] = True

    def take_tiles(self, idx):
        self.tiles_pointers[idx] = False

    def get_pointers(self):
        return self.tiles_pointers

class Player(Handler):
    def __init__(self) -> None:
        Handler.__init__(self)

    def lay_tiles(self, group, tiles_pointers):
        if self.validate_tiles(tiles_pointers):
            group[tiles_pointers] = True
            self.take_tiles(tiles_pointers)

    def validate_tiles(self):
        return True


class Group(Handler):
    def __init__(self) -> None:
        Handler.__init__(self)


if __name__ == '__main__':
    game = Rummikub(2)
    while not game.is_end():
        game.render()
        game.next_move()
