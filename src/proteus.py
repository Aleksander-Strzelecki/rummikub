import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from solver import Solver
from rummikub import Rummikub
import random

class Proteus(object):
    def __init__(self, game: Rummikub, model_path=None) -> None:
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()
        self.game = game

    def get_e_greedy_action(self, state, eps=0.1):
        player = state[0,:]
        groups = state[1:,:]
        any_groups_mask = np.any(groups, axis=1)
        any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]
        valid_board = False

        moves = []
        assessment = []
        ################ GROUP EXTENDING ####################
        for group, idx in zip(any_groups, any_groups_idx):
            tiles_idxs = Solver.solve_no_duplicates(player, group)
            for tile_idx in tiles_idxs:
                any_groups_test = any_groups.copy()
                player_test = player.copy()
                player_test[tile_idx] = False
                any_groups_test[idx, tile_idx] = True
                # Reduce player test and any_groups_test for evaluate_state 
                reduce_groups = np.hstack((any_groups_test[:,:Rummikub.reduced_tiles_number-2] \
                    | any_groups_test[:,Rummikub.reduced_tiles_number-2:Rummikub.tiles_number-2], any_groups_test[:,-2:]))
                reduce_player = np.hstack((player_test[:Rummikub.reduced_tiles_number-2] \
                    | player_test[Rummikub.reduced_tiles_number-2:Rummikub.tiles_number-2], player_test[-2:]))
                assessment.append(self.evaluate_state(np.vstack([reduce_player, reduce_groups])))
                moves.append([-1, idx, tile_idx])
        ################ MOVE FINISH ONLY IF VALID BOARD ####################
        if Solver.check_board(any_groups) and self.game.move_done:
            assessment.append(self.evaluate_state(np.vstack([player, any_groups])))
            moves.append([100,0,0])
            valid_board = True

        moves_array = np.array(moves)
        assessment_array = np.array(assessment)
        chance = random.random()
        if chance < eps and assessment_array.size > 0:
            row = np.random.choice(moves_array.shape[0], 1)
            return moves_array[row[0],:3], np.max(assessment_array)
        if assessment_array.size != 0:
            return moves_array[np.argmax(assessment_array),:], np.max(assessment_array)
        else:
            return np.array([101,0,0]), 0

    def get_action(self, state):
        player = state[0,:]
        groups = state[1:,:]
        any_groups_mask = np.any(groups, axis=1)
        any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]
        valid_board = False

        moves = []
        assessment = []
        ################ GROUP EXTENDING ####################
        for group, idx in zip(any_groups, any_groups_idx):
            tiles_idxs = Solver.solve_no_duplicates(player, group)
            for tile_idx in tiles_idxs:
                any_groups_test = any_groups.copy()
                player_test = player.copy()
                player_test[tile_idx] = False
                any_groups_test[idx, tile_idx] = True
                reduce_groups = np.hstack((any_groups_test[:,:Rummikub.reduced_tiles_number-2] \
                    | any_groups_test[:,Rummikub.reduced_tiles_number-2:Rummikub.tiles_number-2], any_groups_test[:,-2:]))
                reduce_player = np.hstack((player_test[:Rummikub.reduced_tiles_number-2] \
                    | player_test[Rummikub.reduced_tiles_number-2:Rummikub.tiles_number-2], player_test[-2:]))
                assessment.append(self.evaluate_state(np.vstack([reduce_player, reduce_groups])))
                moves.append([-1, idx, tile_idx])
        ################ MOVE FINISH ONLY IF VALID BOARD ####################
        if Solver.check_board(any_groups) and self.game.move_done:
            # assessment.append(self.evaluate_state(np.vstack([player, any_groups])))
            # moves.append([100,0,0])
            valid_board = True

        moves_array = np.array(moves)
        assessment_array = np.array(assessment)
        if valid_board:
            return np.array([100,0,0]), np.max(assessment_array)
        if assessment_array.size != 0:
            return moves_array[np.argmax(assessment_array),:], np.max(assessment_array)
        else:
            return np.array([101,0,0]), 0

    def evaluate_state(self, state):
        return self.model(tf.expand_dims(state, 0)).numpy()[0,0]

    def evaluate_full_state(self, state):
        player = state[0,:]
        groups = state[1:,:]
        any_groups_mask = np.any(groups, axis=1)
        any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]

        return self.evaluate_state(np.vstack([player, any_groups]))

    def update_batch(self, dataset):
        x_train, y_train = dataset
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=1)

    def save_model(self, model_path):
        self.model.save(model_path)
    
    def _build_model(self):
        input_dim = Rummikub.reduced_tiles_number
        self.batch_size = 4
        units = 32
        output_size = 1

        my_lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
        model = keras.models.Sequential(
        [
            my_lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size, activation='linear'),
        ]
        )

        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(
            loss='mean_squared_error',
            optimizer=opt,
            metrics=["accuracy"],
        )
        # model.summary()

        return model