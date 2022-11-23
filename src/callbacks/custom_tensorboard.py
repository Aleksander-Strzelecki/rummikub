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

    def on_epoch_end(self, epoch, logs=None):
        tbv.tensorboard_total_epoch += 1
        tf.summary.scalar('loss', data=logs['loss'], step=tbv.tensorboard_total_epoch)
        tf.summary.scalar('potential_tiles_laid_number', data=tbv.tensorboard_tiles_laid, step=tbv.tensorboard_total_epoch)
        for buffer_name in tbv.tensorboard_buffer_elements:
            tf.summary.scalar('elements in buffer_' + buffer_name, data=tbv.tensorboard_buffer_elements[buffer_name],
            step=tbv.tensorboard_total_epoch)
            wandb.log({'elements in buffer_' + buffer_name: tbv.tensorboard_buffer_elements[buffer_name]})

        total_tiles_counter = 0
        for player_number in tbv.tensorboard_player_tiles_counter:
            tf.summary.scalar('player_tiles_counter_' + str(player_number), data=tbv.tensorboard_player_tiles_counter[player_number],
            step=tbv.tensorboard_total_epoch)
            wandb.log({'player_tiles_counter_' + str(player_number): tbv.tensorboard_player_tiles_counter[player_number]})
            total_tiles_counter += tbv.tensorboard_player_tiles_counter[player_number]
        tf.summary.scalar('total_tiles_counter', data=total_tiles_counter, step=tbv.tensorboard_total_epoch)
        wandb.log({'total_tiles_counter': total_tiles_counter})
        
        for player_number in tbv.tensorboard_manipulation_counter_player:
            tf.summary.scalar('player_manipulation_counter_' + str(player_number), data=tbv.tensorboard_manipulation_counter_player[player_number],
            step=tbv.tensorboard_total_epoch)
            wandb.log({'player_manipulation_counter_' + str(player_number): tbv.tensorboard_manipulation_counter_player[player_number]})
            tbv.tensorboard_manipulation_counter_player[player_number]=0

            tf.summary.scalar('player_reliable_manipulation_counter_' + str(player_number), data=tbv.tensorboard_reliable_manipulation_counter_player[player_number],
            step=tbv.tensorboard_total_epoch)
            wandb.log({'player_reliable_manipulation_counter_' + str(player_number): tbv.tensorboard_reliable_manipulation_counter_player[player_number]})
            tbv.tensorboard_reliable_manipulation_counter_player[player_number]=0

            tf.summary.scalar('player_fake_manipulation_counter_' + str(player_number), data=tbv.tensorboard_fake_manipulation_counter_player[player_number],
            step=tbv.tensorboard_total_epoch)
            wandb.log({'player_fake_manipulation_counter_' + str(player_number): tbv.tensorboard_fake_manipulation_counter_player[player_number]})
            tbv.tensorboard_fake_manipulation_counter_player[player_number]=0

                
        tf.summary.scalar('manipulation_counter', data=tbv.tensorboard_manipulation_counter, step=tbv.tensorboard_total_epoch)
        wandb.log({'manipulation_counter': tbv.tensorboard_manipulation_counter})
        tf.summary.scalar('reliable_manipulation_counter', data=tbv.tensorboard_reliable_manipulation, step=tbv.tensorboard_total_epoch)
        wandb.log({'reliable_manipulation_counter': tbv.tensorboard_reliable_manipulation})
        tf.summary.scalar('fake_manipulation_counter', data=tbv.tensorboard_fake_manipulation, step=tbv.tensorboard_total_epoch)
        wandb.log({'fake_manipulation_counter': tbv.tensorboard_fake_manipulation})
        tbv.tensorboard_manipulation_counter=0
        tbv.tensorboard_fake_manipulation=0
        tbv.tensorboard_reliable_manipulation=0
        
        wandb.log({'loss': logs['loss'], 'potential_tiles_laid_number': tbv.tensorboard_tiles_laid})
        wandb.log({'total_epoch': tbv.tensorboard_total_epoch})

        wandb.config.total_epoch = tbv.tensorboard_total_epoch

