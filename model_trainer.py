"""This module contain the tools to train a model"""

import time
import numpy as np
import tensorflow as tf
from model_handler import ModelHandler

class ModelTrainer(ModelHandler):
    """ModelTrainer can train a model from its operations (training_op, loss, accuracy)"""

    def __init__(self, model_saver, model_builder, epoch=0, best_loss=np.inf):
        """Creates the model trainer

        It's possible to specify a starting epoch or a starting best loss.
        It can be usefull when retrieving a previously trained model
        """

        super().__init__(model_saver, model_builder)

        self._train_init_op, self._test_init_op = self._model_builder.get_init_ops()

        self._training_op, self._loss, self._accuracy = self._model_builder.get_training_ops()

        self._epoch = epoch
        self._best_loss = best_loss

        self._init = tf.global_variables_initializer()

    def train(self, n_epochs, countdown=20):
        """Trains the model over n_epochs epochs

        Uses early stopping and computes the validation accuracy at each epoch.
        To disable early stopping, set countdown parameter to np.inf
        """

        if self._model_saver is not None:
            print("File will be saved in file {} at each progression".format(
                self._model_saver.get_savefile_name()))
        else:
            print("No ModelSaver instance provided. Be careful, the training will not be saved !")

        max_countdown = countdown

        # Dirty version of: "If we did not restore the model from a checkpoint file"
        if self._epoch == 0:
            self._sess.run(self._init)

        while self._epoch < n_epochs:
            if countdown <= 0:
                print("Early stopping")
                break

            batch_loss = []
            batch_accuracy = []

            tick = time.time()

            self._sess.run(self._train_init_op)
            # Foreach batches
            while True:
                # Training time
                try:
                    loss_val, accuracy_val, _ = self._sess.run(
                        [self._loss, self._accuracy, self._training_op])
                    batch_loss.append(loss_val)
                    batch_accuracy.append(accuracy_val)

                except tf.errors.OutOfRangeError:

                    batch_loss = np.mean(batch_loss)
                    batch_accuracy = np.mean(batch_accuracy)

                    if batch_loss < self._best_loss:
                        self._best_loss = batch_loss
                        countdown = max_countdown

                        if self._model_saver is not None:
                            self._saver.save(self._sess, self._model_saver.get_savefile_name())
                            self._model_saver.save_model_json(self._model_builder, self._epoch,
                                                              self._best_loss)
                    else:
                        countdown -= 1

                    # Inference time
                    self._sess.run(self._test_init_op)

                    validation_accuracy = []
                    while True:
                        try:
                            validation_acc_val = self._sess.run([self._accuracy])
                            validation_accuracy.append(validation_acc_val)
                        except tf.errors.OutOfRangeError:
                            # End of inference
                            validation_accuracy = np.mean(validation_accuracy)

                            print("Epoch {}/{} ; Batch loss: {:06f} ; Best loss: {:06f} ; Batch accuracy: {:05.2f}% ; Test accuracy: {:05.2f}% ; Time: {}s"
                                .format(self._epoch, n_epochs, batch_loss, self._best_loss, batch_accuracy*100, validation_accuracy*100, int(time.time()-tick)))

                            break
                    break
                self._epoch = self._epoch + 1

        print("{}/{} epochs made. Stop training".format(self._epoch, n_epochs))
