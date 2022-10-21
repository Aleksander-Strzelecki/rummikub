import numpy as np
from collections import defaultdict
from solver import Solver
import tensorflow as tf
from tensorflow import keras
from rummikub import Rummikub
from dataset import DataSet

class RolloutState():
    @staticmethod
    def get_rollout_state(state):
        player_or_group = np.zeros((state.shape[0],1), dtype=bool)
        player_or_group[0] = True

        return np.expand_dims(np.hstack([player_or_group, state]), axis=0)

class MonteCarloSearchTreeState():
    def __init__(self, state, accepted=False, move_done=False):
        self.state = state
        self._accepted = accepted
        self.no_moves = False
        self.move_done = move_done

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
        any_groups_no_empty_group_idxs = np.where(any_groups_mask)[0]
        any_groups_no_empty_group = groups[any_groups_mask,:]
        if not np.all(any_groups_mask):
            any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation only if place on board
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]
        self.any_groups = any_groups

        moves = []
        ################ GROUP EXTENDING ####################
        for group, group_idx in zip(any_groups, any_groups_idx):
            tiles_idxs = Solver.solve_no_duplicates(player, group)
            for tile_idx in tiles_idxs:
                moves.append([0, group_idx+1, tile_idx])

        ############### GROUP MANIPULATION ##################
        # moves.extend(Solver.solve_manipulation(any_groups_no_empty_group, any_groups, \
        #     any_groups_no_empty_group_idxs+1, any_groups_idx+1))
        
        ################## TABLE VALIDATION ####################
        if Solver.check_board(self.state[1:,:]) and self.move_done:
            moves.append([100, 0, 0])
        elif Solver.check_board(self.state[1:,:]) and not self.move_done:
            moves.append([101,0,0])

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
        # if Solver.check_board(self.state[1:,:]) and not np.any(self.state[0,:]):
            # TODO player tiles laid started from this state
            # return 1
        if self._accepted and self.move_done:
            return 1
        elif self._accepted:
            return 0.2
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
        groups_extended = 0
        move_done = self.move_done
        if from_row == 0:
            reward = 1
            move_done = True
            if np.any(self.state[to_row,:]):
                groups_extended = 0.7
        state_copy = self.state.copy()
        accepted = False
        if from_row < 100:
            state_copy[from_row, tile_idx] = False
            state_copy[to_row, tile_idx] = True
        else:
            accepted = True

        return MonteCarloSearchTreeState(state_copy, accepted=accepted, move_done=move_done), reward, groups_extended

    def get_state(self):
        state = np.vstack([self.state[0,:], self.any_groups])
        reduce_state = np.hstack((state[:,:Rummikub.reduced_tiles_number-2] \
                    | state[:,Rummikub.reduced_tiles_number-2:Rummikub.tiles_number-2], state[:,-2:]))
        player_or_group = np.zeros((reduce_state.shape[0],1), dtype=bool)
        player_or_group[0] = True

        return np.expand_dims(np.hstack([player_or_group, reduce_state]), axis=0)


class MonteCarloTreeSearchNode():

    player_tiles_less = 0
    state_estimate_model = None
    groups_estimate_model = None
    BUFFER_SIZE = 100

    def __init__(self, state: MonteCarloSearchTreeState, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._groups_extended = 0
        self._results_accepted = defaultdict(int)
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
        next_state, reward, reward_train = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)
        self._results[child_node] = reward
        self._groups_extended = reward_train

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        counter = 0
        
        while not current_rollout_state.is_game_over() and counter < 20:
            possible_moves = current_rollout_state.get_legal_actions()
            if possible_moves == []:
                break
            
            action = self.rollout_policy(possible_moves, current_rollout_state)
            current_rollout_state, _, _ = current_rollout_state.move(action)
            counter += 1

        return current_rollout_state.game_result()

    def backpropagate(self, result, child=None, propagated_reward=0, dataset:DataSet=None):
        if dataset is None:
            dataset = DataSet()

        self._number_of_visits += 1.
        if child and result > 0:
            self._results_accepted[child] = self._results[child] + propagated_reward
            propagated_reward = max(self._results_accepted.values())
            self._results_accepted[child] = max(self._results_accepted[child], result)
        elif child is None:
            self._results_accepted[self] = result
        result_train = self._get_result_train()
        dataset.extend_dataset(RolloutState.get_rollout_state(self.state.state), np.array([[1/(1 + np.exp(3-0.5*result_train))]]))
        if self.parent:
            self.parent.backpropagate(result, child=self, propagated_reward=propagated_reward, dataset=dataset)
        else:
            x_train, y_train = dataset.get_data()
            self.state_estimate_model.fit(x_train, y_train)

        return dataset

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
    
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves, current_rollout_state):
        possible_moves_np = np.array(possible_moves)
        if 100 in possible_moves_np[:,0]:
            return [100, 0, 0]
        possible_rollout_states = self._get_possible_rollout_states(current_rollout_state, possible_moves)
        state_estimation = self.state_estimate_model.predict(possible_rollout_states)
        state_distribution = (np.array(state_estimation) / np.sum(state_estimation)).flatten()

        rng = np.random.default_rng()
        return rng.choice(possible_moves_np, p=state_distribution, axis=0)
        # return possible_moves_np[np.argmax(state_estimation)]

    def _get_possible_rollout_states(self, current_rollout_state, possible_moves):
        possible_rollout_states = []
        for move in possible_moves:
            from_row, to_row, tile_idx = move[0], move[1], move[2]
            state_copy = current_rollout_state.state.copy()
            if from_row < 100:
                state_copy[from_row, tile_idx] = False
                state_copy[to_row, tile_idx] = True
            possible_rollout_states.extend(RolloutState.get_rollout_state(state_copy))

        return np.array(possible_rollout_states)

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def _get_result_train(self):
        if self._results_accepted:
            return max(max(self._results_accepted.values()), self._groups_extended)
        else:
            return self._groups_extended

    def best_action(self):
        simulation_no = 150
        dataset = DataSet()
        
        for i in range(simulation_no):
            
            v = self._tree_policy()
            reward = v.rollout()
            dataset = v.backpropagate(reward, dataset=dataset)
            dataset.shrink(self.BUFFER_SIZE)
        
        return self.best_child(c_param=0.)

    @classmethod
    def create_models(cls):
        cls._build_groups_estimate_model()
        cls._build_state_estimate_model()

    @classmethod
    def _build_state_estimate_model(cls):
        input_dim = Rummikub.tiles_number+1 # one bit true if player false if group
        cls.batch_size = 4
        units = 32
        output_size = 1

        model = keras.models.Sequential(
        [
            keras.layers.Bidirectional(keras.layers.LSTM(units), \
                input_shape=(None, input_dim)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size, activation='sigmoid'),
        ]
        )

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=["accuracy"],
        )

        cls.state_estimate_model = model

    @classmethod
    def _build_groups_estimate_model(cls):
        input_dim = Rummikub.tiles_number+1 # one bit true if player false if group
        cls.batch_size = 4
        units = 32
        output_size = 1

        model = keras.models.Sequential(
        [
            keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=True), \
                input_shape=(None, input_dim)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size, activation='sigmoid'),
        ]
        )

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=["accuracy"],
        )

        cls.groups_estimate_model = model