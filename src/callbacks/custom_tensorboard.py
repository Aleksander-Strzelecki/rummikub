import tensorflow as tf
import global_variables.tensorboard_variables as tbv
from tensorflow import keras
from datetime import datetime

class CustomTensorboard(tf.keras.callbacks.Callback):
    path_prefix = ''
    logdir = path_prefix + "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    total_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.total_epoch += 1
        tf.summary.scalar('loss', data=logs['loss'], step=self.total_epoch)
        if tbv.tensorboard_move_done:
            tf.summary.scalar('loss_when_move', data=logs['loss'], step=self.total_epoch)
            tf.summary.scalar('tiles_laid', data=tbv.tensorboard_tiles_laid, step=self.total_epoch)
        else:
            tf.summary.scalar('loss_when_no_move', data=logs['loss'], step=self.total_epoch)
