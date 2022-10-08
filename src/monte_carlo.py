import numpy as np
from collections import defaultdict
from solver import Solver

class MonteCarloSearchTreeState():
    def __init__(self, state):
        self.state = state
        self._reward = 0

    def get_legal_actions(self): 
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        player = self.state[0,:]
        groups = self.state[1:,:]
        any_groups_mask = np.any(groups, axis=1)
        any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]

        moves = []
        ################ GROUP EXTENDING ####################
        for group, group_idx in zip(any_groups, any_groups_idx):
            tiles_idxs = Solver.solve_no_duplicates(player, group)
            for tile_idx in tiles_idxs:
                moves.append([0, group_idx, tile_idx])

        return moves

    def is_game_over(self):
        '''
        Modify according to your game or 
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        return not np.any(self.state[0,:])

    def game_result(self):
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        if Solver.check_board(self.state[1:,:]) and not np.any(self.state[0,:]):
            # TODO player tiles laid started from this state
            return 1
        return 0

    def move(self,action):
        '''
        Modify according to your game or 
        needs. Changes the state of your 
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board 
        position is empty. If you place x in
        row 2 column 3, then it would be some 
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns 
        the new state after making a move.
        '''
        from_row, to_row, tile_idx = action[0], action[1], action[2]
        state_copy = self.state.copy()
        state_copy[from_row, tile_idx] = False
        state_copy[to_row, tile_idx] = True

        return MonteCarloSearchTreeState(state_copy)


class MonteCarloTreeSearchNode():

    player_tiles_less = 0

    def __init__(self, state: MonteCarloSearchTreeState, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        # self._results = {}
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):

        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    # TODO Modify q for my purpose
    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
	
        action = self._untried_actions.pop()
        next_state, _ = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)

        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
    
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
    
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 100
        
        
        for i in range(simulation_no):
            
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)