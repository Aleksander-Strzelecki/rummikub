import tensorflow as tf
import global_variables.tensorboard_variables as tbv
from tensorflow import keras
from datetime import datetime
import wandb

class CustomTensorboard(tf.keras.callbacks.Callback):
    path_prefix = ''
    logdir = path_prefix + "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics", max_queue=100)
    file_writer.set_as_default()
    total_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.total_epoch += 1
        tf.summary.scalar('loss', data=logs['loss'], step=self.total_epoch)
        tf.summary.scalar('tiles_laid', data=tbv.tensorboard_tiles_laid, step=self.total_epoch)
        for buffer_name in tbv.tensorboard_buffer_elements:
            tf.summary.scalar('elements in buffer_' + buffer_name, data=tbv.tensorboard_buffer_elements[buffer_name],
            step=self.total_epoch)
            wandb.log({'elements in buffer_' + buffer_name: tbv.tensorboard_buffer_elements[buffer_name]})

        for player_number in tbv.tensorboard_player_tiles_counter:
            tf.summary.scalar('player_tiles_counter_' + str(player_number), data=tbv.tensorboard_player_tiles_counter[player_number],
            step=self.total_epoch)
            wandb.log({'player_tiles_counter_' + str(player_number): tbv.tensorboard_player_tiles_counter[player_number]})
        for player_number in tbv.tensorboard_manipulation_counter_player:
            tf.summary.scalar('player_manipulation_counter_' + str(player_number), data=tbv.tensorboard_manipulation_counter_player[player_number],
            step=self.total_epoch)
            wandb.log({'player_manipulation_counter_' + str(player_number): tbv.tensorboard_manipulation_counter_player[player_number]})

        tf.summary.scalar('manipulation_counter', data=tbv.tensorboard_manipulation_counter, step=self.total_epoch)

        wandb.log({'loss': logs['loss'], 'tiles_laid': tbv.tensorboard_tiles_laid})

