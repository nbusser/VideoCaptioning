import tensorflow as tf

class ModelHandler(object):
    def __init__(self, model_saver, model_builder):
        """Creates the model handler

        In addition to the model builder, creating a ModelTrainer requires
        a ModelSaver instance. It contains all meta-parameters of the  model
        and manages to save the model or to restore it from a checkpoint
        """

        self._model_saver = model_saver
        self._model_builder = model_builder

        self._sess = tf.Session()

        self._saver = tf.train.Saver()

    def load_last_checkpoint(self):
        """This function restores the checkpoint associated to the ModelSaver"""
        self._saver.restore(self._sess, self._model_saver.get_savefile_name())

    def close_session(self):
        """Closes the tf.Session()"""
        self._sess.close()
