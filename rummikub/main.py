import numpy as np

class Rummikub:
    def __init__(self, num_players) -> None:
        self.players = []
        for i in range(num_players):
            self.players.append(Player())
        self.tiles_pointers = np.ones((104), dtype=bool)
        self.tiles = {}
        self.create_tiles()
        self.distribute_tiles()
        

    def create_tiles(self):
        colors = ['r', 'b', 'o', 'b']
        counter = 0
        for color in colors:
            for i in range(13):
                self.tiles[counter] = (color, i)
                counter += 1
        self.tiles[counter] = ('j', 0)
        counter += 1
        self.tiles[counter] = ('j', 0)

    def distribute_tiles(self):
        indexes = np.random.choice(104, 14*len(self.players))
        for idx, player in enumerate(self.players):
            player.set_tiles(indexes[14*idx:14*(idx+1)])
            self.tiles_pointers[indexes[14*idx:14*(idx+1)]] = False

    def render(self):
        print(self.tiles_pointers)
        for player in self.players:
            print(player.tiles_pointers)
            # for idx in np.argwhere(self.tiles_pointers == True).tolist():
            #     print(self.tiles[idx])

class Player(object):
    def __init__(self) -> None:
        self.rack = Rack()
        self.tiles_pointers = np.zeros((104), dtype=bool)

    def set_tiles(self, idx):
        self.tiles_pointers[idx] = True

    def __str__(self) -> str:
        print(self.tiles_pointers)

class Rack(object):
    def __init__(self) -> None:
        self.tiles = []


if __name__ == '__main__':
    game = Rummikub(2)
    game.render()
