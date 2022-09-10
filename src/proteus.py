import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from solver import Solver
from rummikub import Rummikub
import random

class Proteus(object):
    def __init__(self) -> None:
        self.model = self._build_model()
        self.model.compile(
            loss='mean_squared_error',
            optimizer="adam",
            metrics=["accuracy"],
        )

    def get_e_greedy_action(self, state, eps=0.1):
        player = state[0,:]
        groups = state[1:,:]
        any_groups_mask = np.any(groups, axis=1)
        any_groups_mask[np.where(any_groups_mask==False)[0][0]] = True  # add one empty group to evaluation
        any_groups_idx = np.where(any_groups_mask)[0]
        any_groups = groups[any_groups_mask,:]

        moves = []
        assessment = []
        ################ GROUP EXTENDING ####################
        for group, idx in zip(any_groups, any_groups_idx):
            tiles_idxs = Solver.solve_pair(player, group)
            for tile_idx in tiles_idxs:
                any_groups_test = any_groups.copy()
                player_test = player.copy()
                player_test[tile_idx] = False
                any_groups_test[idx, tile_idx] = True
                assessment.append(self.evaluate_state(np.vstack([player_test, any_groups_test])))
                moves.append([-1, idx, tile_idx])
        ################ MOVE FINISH ####################
        assessment.append(self.evaluate_state(np.vstack([player, any_groups])))
        moves.append([100,0,0])

        moves_array = np.array(moves)
        assessment_array = np.array(assessment)
        if max(assessment) < 0.01:
            return [101, 0, 0]
        if random.random() < eps:
            row = np.random.choice(moves_array.shape[0], 1)
            return moves_array[row[0],:3]
        return moves_array[np.argmax(assessment_array),:]

    def evaluate_state(self, state):
        return self.model(tf.expand_dims(state, 0)).numpy()[0,0]

    def _build_model(self):
        input_dim = Rummikub.tiles_number
        batch_size = 1
        units = 64
        output_size = 1

        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
        model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size, activation='relu'),
        ]
        )
        # model.summary()

        return model
