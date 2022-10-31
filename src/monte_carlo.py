import numpy as np
import tensorflow as tf
import os
import global_variables.tensorboard_variables as tbv

from tensorflow import keras
from rummikub import Rummikub
from dataset import DataSet
from callbacks.custom_tensorboard import CustomTensorboard
from collections import defaultdict
from solver import Solver

class StateANN():
    @staticmethod
    def get_state_ann(state):
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
        solver = Solver(groups)
        ################ GROUP EXTENDING ####################
        moves.extend(solver.solve_player_groups(player, offset=1))
        ############### GROUP MANIPULATION ##################
        moves.extend(solver.solve_manipulation(offset=1))
        
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
    model_checkpoint_callback = None
    model_custom_tensorboard_callback = None
    BUFFER_SIZE = 50
    POSITIVE_BUFFER_SIZE = 500

    def __init__(self, state: MonteCarloSearchTreeState, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 1
        self._results = defaultdict(int)
        self._groups_extended = 0
        self._results_accepted = defaultdict(int)
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self._untried_states_ann = self._get_possible_states(self.state, self._untried_actions)
        return

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    # TODO Modify q for my purpose
    def q(self):
        if self._results_accepted:
            return max(self._results_accepted.values())
        return 0

    def n(self):
        return self._number_of_visits

    def expand(self):
	
        for action in self._untried_actions:
            next_state, reward, group_extended_reward = self.state.move(action)
            child_node = MonteCarloTreeSearchNode(
                next_state, parent=self, parent_action=action)
            self._results[child_node] = reward
            self._groups_extended = max(self._groups_extended, group_extended_reward)
            self.children.append(child_node)

        action = self._get_probable_untried_action()
        
        return self.children[self._untried_actions.index(action)]

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        counter = 0
        actions_to_return = None
        actions_table = []

        current_node = self
        actions_table.append(current_node.parent_action)
        current_node = current_node.parent
        while current_node.parent:
            actions_table.insert(0, current_node.parent_action)
            current_node = current_node.parent
        
        while not current_rollout_state.is_game_over() and counter < 20:
            possible_moves = current_rollout_state.get_legal_actions()
            if possible_moves == []:
                break
            
            action = self.rollout_policy(possible_moves, current_rollout_state)
            current_rollout_state, _, _ = current_rollout_state.move(action)
            counter += 1

            actions_table.append(action.tolist())
        
        if actions_table[-1] == [100,0,0]:
            actions_to_return = actions_table

        return current_rollout_state.game_result(), actions_to_return

    def backpropagate(self, result, child=None, propagated_reward=0, dataset:DataSet=None, positive_dataset:DataSet=None):
        self._number_of_visits += 1.
        if child and result > 0:
            self._results_accepted[child] = self._results[child] + propagated_reward
            propagated_reward = max(self._results_accepted.values())
            self._results_accepted[child] = max(self._results_accepted[child], result)
        elif child is None:
            self._results_accepted[self] = result
        result_train = self._get_result_train()
        self._extend_datasets(dataset, positive_dataset, result_train)
        if self.parent:
            self.parent.backpropagate(result, child=self, propagated_reward=propagated_reward,
                dataset=dataset, positive_dataset=positive_dataset)
        else:
            x_train, y_train = dataset.get_data()
            x_train_positive, y_train_positive = positive_dataset.get_data()
            x_concatenate = np.concatenate((x_train, x_train_positive), axis=0) if x_train_positive.size else x_train
            y_concatenate = np.concatenate((y_train, y_train_positive), axis=0) if y_train_positive.size else y_train
            self._fit_model_with_callbacks(x_concatenate, y_concatenate, result, propagated_reward)

        return dataset, positive_dataset

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.3, ann_param=1.0, verbose=False):
        children_estimation_ann = self.state_estimate_model.predict(self._untried_states_ann, verbose=0)

        choices_weights = [c.q() + c_param * ((np.log(self.n()) / (c.n() + 1))) + ann_param * ann_estimation\
             for c, ann_estimation in zip(self.children, children_estimation_ann)]
        if verbose:
            print(self._results_accepted)
            print(self.children)
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves, current_rollout_state):
        possible_moves_np = np.array(possible_moves)
        if 100 in possible_moves_np[:,0]:
            return np.array([100, 0, 0])
        possible_rollout_states = self._get_possible_states(current_rollout_state, possible_moves)
        state_distribution = self._get_state_distribution(possible_rollout_states)

        return self._get_action_from_distribution(possible_moves_np, state_distribution)

    def _get_possible_states(self, current_rollout_state, possible_moves):
        possible_rollout_states = []
        for move in possible_moves:
            from_row, to_row, tile_idx = move[0], move[1], move[2]
            state_copy = current_rollout_state.state.copy()
            if from_row < 100:
                state_copy[from_row, tile_idx] = False
                state_copy[to_row, tile_idx] = True
            possible_rollout_states.extend(StateANN.get_state_ann(state_copy))

        return np.array(possible_rollout_states)

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if current_node._no_children():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def _get_result_train(self):
        if self._results_accepted:
            return max(max(self._results_accepted.values()), self._groups_extended)
        else:
            return self._groups_extended

    def best_actions(self, buffer:DataSet, positive_buffer:DataSet):
        simulation_no = 9
        actions = []
        spare_actions = []

        for i in range(simulation_no):
            #TODO save success actions from rollout
            v = self._tree_policy()
            reward, rollout_actions = v.rollout()
            if rollout_actions:
                spare_actions.append(rollout_actions)
            buffer, positive_buffer = v.backpropagate(reward, dataset=buffer, positive_dataset=positive_buffer)
            buffer.shrink(self.BUFFER_SIZE)
            positive_buffer.shrink(self.POSITIVE_BUFFER_SIZE)
        
        child = self.best_child(c_param=0., ann_param=0., verbose=True)
        actions.append(child.parent_action)
        while child.children:
            child = child.best_child(c_param=0., ann_param=0., verbose=True)
            print('Parent action:' + str(child.parent_action))
            actions.append(child.parent_action)

        if actions[-1] == [100,0,0]:
            print("Action from monte carlo search")
            return actions, buffer
        elif spare_actions:
            print('Action from rollout')
            return spare_actions[-1], buffer
        else:
            print('Action not found return default')
            return [[101,0,0]], buffer

    def _get_probable_untried_action(self):
        state_distribution = self._get_state_distribution(self._untried_states_ann)
        action = self._get_action_from_distribution(np.array(self._untried_actions), state_distribution)
        action = action.tolist()

        return action

    def _get_state_distribution(self, possible_states):
        state_estimation = self.state_estimate_model.predict(possible_states, verbose=0)
        state_distribution = (np.array(state_estimation) / np.sum(state_estimation)).flatten()

        return state_distribution

    def _get_action_from_distribution(self, actions:np.ndarray, distribution):
        rng = np.random.default_rng()
        return rng.choice(actions, p=distribution, axis=0)

    def _no_children(self):
        return self.children == []

    def _extend_datasets(self, dataset:DataSet, positive_dataset:DataSet, train_reward):
        reward_function_return = self._reward_function(train_reward)
        if train_reward > 1:
            positive_dataset.extend_dataset(StateANN.get_state_ann(self.state.state), reward_function_return)
        dataset.extend_dataset(StateANN.get_state_ann(self.state.state), reward_function_return)

    def _reward_function(self, reward):
        return np.array([[1/(1 + np.exp(5-3*reward))]])

    def _fit_model_with_callbacks(self, x_train, y_train, result, propagated_reward):
        if result > 0.9:
            tbv.tensorboard_move_done = True
            tbv.tensorboard_tiles_laid = propagated_reward
        else:
            tbv.tensorboard_tiles_laid = 0
            tbv.tensorboard_move_done = False
        self.state_estimate_model.fit(x_train,
                 y_train, callbacks=[self.model_checkpoint_callback, self.model_custom_tensorboard_callback])

    @classmethod
    def create_models(cls):
        cls._build_state_estimate_model()

    @classmethod
    def _build_state_estimate_model(cls):
        input_dim = Rummikub.tiles_number+1 # one bit true if player false if group
        cls.batch_size = 4
        units = 64
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
        )

        checkpoint_filepath = './models/checkpoint'
        cls.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_freq=10)

        cls.model_custom_tensorboard_callback = CustomTensorboard()
        
        if os.path.isfile(checkpoint_filepath): 
            model.load_weights(checkpoint_filepath)
        cls.state_estimate_model = model
        