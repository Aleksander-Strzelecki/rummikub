from termcolor import colored
import numpy as np
from itertools import groupby
import os

class Rummikub:
    # control numbers:
    # 100 - end of move
    # 101 - get new tile
    def __init__(self, num_players) -> None:
        self.activ = 0
        self.num_players = num_players
        self.players = np.zeros((num_players, 106), dtype=bool)
        # for i in range(num_players):
        #     self.players.append(Player())
        self.tiles_pointers = np.ones((106), dtype=bool)
        self.tiles = {}
        self.create_tiles()
        self.distribute_tiles()
        self.groups = np.zeros((36, 106), dtype=bool)
        # self.create_groups()
        self.groups_backup = self.groups.copy()
        self.players_backup = self.players.copy()
        self.colors_pointers = {0:'magenta', 1:'red', 2:'white', 3:'yellow', 4:'blue'}
        
    def create_tiles(self):
        colors = [1,1,2,2,3,3,4,4]
        counter = 0
        for color in colors:
            for i in range(1,14):
                self.tiles[counter] = (color, i)
                counter += 1
        self.tiles[counter] = (0, 0)
        counter += 1
        self.tiles[counter] = (0, 0)

    def distribute_tiles(self):
        indexes = np.random.choice(106, 14*len(self.players), replace=False)
        # print(len(indexes))
        for idx in range(self.num_players):
            choice = indexes[14*idx:14*(idx+1)]
            # print(choice, len(choice))
            # player.set_tiles(choice)
            self.players[idx, choice] = True
            self.tiles_pointers[choice] = False

    def render(self):
        os.system('clear')
        
        tiles_idx = self.get_true_idx(self.tiles_pointers)
        print("Free tiles[{}]: ".format(len(tiles_idx)))
        self.print_tiles(tiles_idx)

        for idx in range(self.num_players):
            tiles_idx = self.get_true_idx(self.players[idx,:])
            if self.activ == idx:
                print('->', end='')
            print(f"Player {idx}[{len(tiles_idx)}]: ")
            self.print_tiles(tiles_idx)

        groups_idx = np.unique(self.get_true_idx(self.groups))
        for number, group_idx in enumerate(groups_idx):
            tiles_idx = self.get_true_idx(self.groups[group_idx,:])
            print(f"Group {group_idx}[{len(tiles_idx)}]: ")
            self.print_tiles(tiles_idx)

    def get_true_idx(self, array):
        return np.nonzero(array)[0]

    def print_tiles(self, tiles_idx):
        for t_idx in tiles_idx:
            tile = self.tiles[t_idx]
            print(colored(tile[1], self.colors_pointers[tile[0]]), '(' + str(t_idx) + ')', end=" ")
        print('')

    # def create_groups(self):
    #     for i in range(36):
    #         self.groups.append(np.zeros((106), dtype=bool))

    def next_move(self):
        # actual_player = self.players[self.activ]
        from_group = int(input("From: "))
        if from_group < 100:
            to_group = int(input("To: "))
            t_pointer = int(input())
            
            target = self.groups[to_group,:]
            if self.validate_move(target, t_pointer):
                if from_group == -1:
                    # actual_player.take_tiles(t_pointer)
                    self.players[self.activ, t_pointer] = False
                else:
                    self.groups[from_group,t_pointer] = False
                self.groups[to_group,t_pointer] = True
        
        elif from_group == 100:
            if self.validate_board():
                self.activ = (self.activ + 1) % self.num_players
                self._commit()
            else:
                self._rollback()
        elif from_group == 101:
            self._rollback()
            selected_tiles = np.random.choice(self.get_true_idx(self.tiles_pointers))
            self.players[self.activ, selected_tiles] = True
            self.tiles_pointers[selected_tiles] = False
            self.activ = (self.activ + 1) % self.num_players
            self._commit()

    def validate_move(self, target, t_pointer):
        target[t_pointer] = True
        result = self.check_group(target)
        target[t_pointer] = False
        return result

    def validate_board(self):
        count = np.sum(self.groups, axis=1)
        count_no_zero = np.where(count==0, 3, count)
        return np.all(count_no_zero > 2)

        

    def check_group(self, target):
        tiles_idx = self.get_true_idx(target)
        if len(tiles_idx) == 1:
            return True
        
        colors = []
        numbers = []
        for idx in tiles_idx:
            colors.append(self.tiles[idx][0])
            numbers.append(self.tiles[idx][1])

        if self.all_equal(colors) and self.checkConsecutive(numbers):
            return True
        if self.all_equal(numbers) and self.checkUnique(colors):
            return True
        return False

    def all_equal(self, iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    def checkUnique(self, l):
        return np.unique(l).size == len(l)

    def checkConsecutive(self, l):
        n = len(l) - 1
        return (sum(np.diff(sorted(l)) == 1) >= n)

    def is_end(self):
        return False

    def _commit(self):
        self.groups_backup = self.groups.copy()
        self.players_backup = self.players.copy()

    def _rollback(self):
        self.groups = self.groups_backup.copy()
        self.players = self.players_backup.copy()

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
