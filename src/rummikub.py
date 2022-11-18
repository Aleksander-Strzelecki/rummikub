from termcolor import colored
import numpy as np
from itertools import groupby
import os

class Rummikub:
    # control numbers:
    # 100 - end of move
    # 101 - get new tile
    tiles = np.zeros((2, 106), dtype=int)
    tiles_number = 106
    reduced_tiles_number = 54
    def __init__(self, num_players, learning=False, path='') -> None:
        self._path=path
        self.activ = 0
        self.move_done = False
        self.num_players = num_players
        self.players = np.zeros((num_players, 106), dtype=bool)
        self.move_score = 0
        self.tiles_pointers = np.ones((106), dtype=bool)
        self.groups = np.zeros((36, 106), dtype=bool)
        self.existing_groups = []
        self.groups_backup = self.groups.copy()
        self.players_backup = self.players.copy()
        self.colors_pointers = {0:'magenta', 1:'red', 2:'white', 3:'yellow', 4:'blue'}
        self._reward = 0
        self._learning = learning
        self.distribute_tiles()
        
    @classmethod
    def create_tiles(cls):
        colors = [1,2,3,4,1,2,3,4]
        counter = 0
        for color in colors:
            for i in range(1,14):
                cls.tiles[:,counter] = [color, i]
                counter += 1
        cls.tiles[:,counter] = [0, 0]
        counter += 1
        cls.tiles[:,counter] = [0, 0]

    def distribute_tiles(self):
        if self._learning:
            tiles_per_player = 106 // len(self.players)
        else:
            tiles_per_player = 14
        indexes = np.random.choice(106, tiles_per_player*len(self.players), replace=False)
        for idx in range(self.num_players):
            choice = indexes[tiles_per_player*idx:tiles_per_player*(idx+1)]
            self.players[idx, choice] = True
            self.tiles_pointers[choice] = False

    def render(self):
        os.system('clear')
        
        print("Actual scores: {}".format(self.move_score))
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

    def get_player_tiles_number(self, player_id):
        tiles_idx = self.get_true_idx(self.players[player_id,:])
        return len(tiles_idx)

    def print_tiles(self, tiles_idx):
        for t_idx in tiles_idx:
            tile = self.tiles[:,t_idx]
            print(colored(tile[1], self.colors_pointers[tile[0]]), '(' + str(t_idx) + ')', end=" ")
        print('')

    def next_move(self, from_group, to_group, t_pointer):
        # from_group = int(input("From: "))
        self._reward = 0
        if from_group < 100:
            # to_group = int(input("To: "))
            # t_pointer = int(input())
            if from_group == -1 and not self.players[self.activ, t_pointer]:
                return self._get_state(), 0
            elif from_group > -1 and not self.groups[from_group, t_pointer]:
                return self._get_state(), 0
            
            target = self.groups[to_group,:]
            if self.validate_move(target, t_pointer):
                self.move_done = True
                if from_group == -1:
                    self.move_score += (self.tiles[1,t_pointer] * 7)
                    self.players[self.activ, t_pointer] = False
                else:
                    self.groups[from_group,t_pointer] = False
                self.groups[to_group,t_pointer] = True
        
        
        elif from_group == 100 and self.move_done:
            if self.validate_board():
                self.activ = (self.activ + 1) % self.num_players
                self._reward = 1
                self._commit()
            else:
                self._reward = 0
                self._rollback()
            self.move_done = False
            self.move_score = 0
        elif from_group == 101 and np.any(self.tiles_pointers):
            self.move_done = False
            self.move_score = 0
            self._reward = 0
            self._rollback()
            selected_tiles = np.random.choice(self.get_true_idx(self.tiles_pointers))
            self.players[self.activ, selected_tiles] = True
            self.tiles_pointers[selected_tiles] = False
            self.activ = (self.activ + 1) % self.num_players
            self._commit()
        elif from_group == 101 and not np.any(self.tiles_pointers):
            self.move_done = False
            self.move_score = 0
            self._reward = 0
            self._rollback()
            self.activ = (self.activ + 1) % self.num_players
            self._commit()

        return self._get_state(), self._get_reward()

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
            self._reward = 0
            return True
        
        colors = self.tiles[0,tiles_idx]
        numbers = self.tiles[1,tiles_idx]

        if self.all_equal(colors) and self.checkConsecutive(numbers) and len(numbers) < 14:
            self._reward = 0.1
            # if len(numbers) == 3:
            #     self._reward = 0.3
            return True
        if self.all_equal(numbers) and self.checkUnique(colors) and len(colors) < 5:
            self._reward = 0.1
            # if len(numbers) == 3:
            #     self._reward = 0.3
            return True
        return False

    def all_equal(self, array):
        array_no_joker = array[array != 0]
        if array_no_joker.size == 0:
            return True
        return np.all(array_no_joker == array_no_joker[0])

    def checkUnique(self, array):
        array_no_joker = array[array != 0]
        return np.unique(array_no_joker).size == len(array_no_joker)

    def checkConsecutive(self, array):
        jokers_count = len(array[array == 0])
        if jokers_count == 0:
            return np.all(np.diff(np.sort(array)) == 1)    
        array_no_joker = array[array != 0]
        diff_array = np.diff(np.sort(array_no_joker))
        if np.sum(diff_array // 2) <= jokers_count:
            return np.all(((diff_array%2==0) & (diff_array>0)) | (diff_array==1))
        return False

    def reset(self):
        self.activ = 0
        self.move_done = False
        self.players.fill(False)
        self.move_score = 0
        self.tiles_pointers.fill(True)
        self.distribute_tiles()
        self.groups.fill(False)
        self.groups_backup = self.groups.copy()
        self.players_backup = self.players.copy()
        self._reward = 0

        return self._get_state()

    def save_state(self):
        path = self._path + 'rummikub_state.npy'
        with open(path, 'wb') as f:
            print("Saving rummikub state to {}".format(path))
            np.save(f, self.players)
            np.save(f, self.groups)
            np.save(f, self.tiles_pointers)

    def load_state(self):
        path = self._path + 'rummikub_state.npy'
        isExist = os.path.exists(path)
        if isExist:
            with open(path, 'rb') as f:
                print("Loading rummikub state from {}".format(path))
                self.players = np.load(f)
                self.groups = np.load(f)
                self.tiles_pointers = np.load(f)
            self.activ = 0
            self.move_done = False
            self.move_score = 0
            self.groups_backup = self.groups.copy()
            self.players_backup = self.players.copy()
            self._reward = 0

        return self._get_state()

    
    def is_end(self):
        return not np.any(self.players[(self.activ-1) % self.num_players, :])

    def _commit(self):
        self.groups_backup = self.groups.copy()
        self.players_backup = self.players.copy()
        self.existing_groups = self.get_true_idx(self.groups)

    def _rollback(self):
        self.groups = self.groups_backup.copy()
        self.players = self.players_backup.copy()

    def _get_state(self):
        player_tiles = self.players[self.activ, :]
        return np.vstack([player_tiles, self.groups])

    def _get_reward(self):
        return self._reward

print("Initializing rummikub tiles")
Rummikub.create_tiles()