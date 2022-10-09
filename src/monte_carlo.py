import numpy as np
from collections import defaultdict
from solver import Solver

class MonteCarloSearchTreeState():
    def __init__(self, state, accepted=False):
        self.state = state
        self._accepted = accepted
        self.no_moves = False

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
        if not np.all(any_groups_mask):
            any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation only if place on board
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]

        moves = []
        ################ GROUP EXTENDING ####################
        for group, group_idx in zip(any_groups, any_groups_idx):
            tiles_idxs = Solver.solve_no_duplicates(player, group)
            for tile_idx in tiles_idxs:
                moves.append([0, group_idx+1, tile_idx])

        ################## TABLE VALIDATION ####################
        if Solver.check_board(self.state[1:,:]):
            moves.append([100, 0, 0])

        if moves == []:
            self.no_moves = True

        return moves

    def is_game_over(self):
        '''
        Modify according to your game or 
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        # end if player has no tiles or no free place on table or after sumbitting actual state of the table
        return self._accepted or self.no_moves or (not np.any(self.state[0,:]))

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
        elif self._accepted:
            return 0.5
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
        reward = 0
        if from_row == 0:
            reward = 1
        state_copy = self.state.copy()
        accepted = False
        if from_row < 100:
            state_copy[from_row, tile_idx] = False
            state_copy[to_row, tile_idx] = True
        else:
            accepted = True

        return MonteCarloSearchTreeState(state_copy, accepted=accepted), reward


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
        self._results_accepted = defaultdict(int)
        # self._results[1] = 0
        # self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):

        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    # TODO Modify q for my purpose
    def q(self):
        return max(self._results_accepted.values())

    def n(self):
        return self._number_of_visits

    def expand(self):
	
        action = self._untried_actions.pop()
        next_state, reward = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)
        self._results[child_node] += reward

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            if possible_moves == []:
                break
            
            action = self.rollout_policy(possible_moves, current_rollout_state)
            current_rollout_state, _ = current_rollout_state.move(action)

        return current_rollout_state.game_result()

    def backpropagate(self, result, child=None, propagated_reward=0):
        self._number_of_visits += 1.
        if child and result > 0:
            self._results_accepted[child] = self._results[child] + propagated_reward
            propagated_reward = max(self._results_accepted.values())
            self._results_accepted[child] = max(self._results_accepted[child], result)
        if self.parent:
            self.parent.backpropagate(result, child=self, propagated_reward=propagated_reward)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
    
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves, current_rollout_state):
    
        possible_moves_np = np.array(possible_moves)
        if 100 in possible_moves_np[:,0]:
            return [100, 0, 0]
        current_rollout_groups = current_rollout_state.state[1:,:]
        # non_empty_current_rollout_groups = current_rollout_groups[np.any(current_rollout_groups, axis=1),:]
        tiles_in_current_rollout_groups = np.sum(current_rollout_groups, axis=1)
        
        double_current_rollout_groups_rows = np.where(tiles_in_current_rollout_groups == 2)[0] + 1
        possible_moves_to_double_groups = possible_moves_np[np.in1d(possible_moves_np[:,1], double_current_rollout_groups_rows) & (possible_moves_np[:,0]==0), :]
        # add tile to group that have already two tiles
        if possible_moves_to_double_groups.size > 0:
            return possible_moves_to_double_groups[np.random.randint(len(possible_moves_to_double_groups))]
                
        single_current_rollout_groups_rows = np.where(tiles_in_current_rollout_groups == 1)[0] + 1
        possible_moves_to_single_groups = possible_moves_np[np.in1d(possible_moves_np[:,1], single_current_rollout_groups_rows) & (possible_moves_np[:,0]==0), :]
        # add tile to group that have already one tile
        if possible_moves_to_single_groups.size > 0:
            return possible_moves_to_single_groups[np.random.randint(len(possible_moves_to_single_groups))]

        non_empty_current_rollout_groups_rows = np.where(tiles_in_current_rollout_groups)[0] + 1
        possible_moves_to_non_empty_groups = possible_moves_np[np.in1d(possible_moves_np[:,1], non_empty_current_rollout_groups_rows) & (possible_moves_np[:,0]==0), :]
        # add tile to group that have already one tile
        if possible_moves_to_non_empty_groups.size > 0:
            return possible_moves_to_non_empty_groups[np.random.randint(len(possible_moves_to_non_empty_groups))]

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
