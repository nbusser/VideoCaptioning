"""This module contais ModelPredictor class, used to do predictions"""

from model_handler import ModelHandler
import tensorflow as tf

class ModelPredictor(ModelHandler):
    """This class is used to do predictions over the model"""

    def __init__(self, model_saver, model_builder):
        super().__init__(model_saver, model_builder)

        self._test_init_op = self._model_builder.get_test_init_op()

        (self._names, self._captions,
         self._predictions, self._accuracy) = self._model_builder.get_testing_ops()

    def predict(self, stop_end=False):
        """Launches and prints predictions over the model

        If stop_end argument is True, the de-tokenization of sentences will stop at the
        first <end> token found
        """

        self._sess.run(self._test_init_op)

        while True:
            try:
                predictions, captions, names, accuracy = self._sess.run(
                    [self._predictions, self._captions, self._names, self._accuracy])

                print("Accuracy {:05.2f}%".format(accuracy*100))

                data_size = len(predictions)
                for i in range(data_size):
                    print("---------------")
                    print(names[i].decode('utf-8'))
                    print("Truth: ", self._from_tokens_to_sentence(captions[i], stop_end))
                    print("Predictions: ", self._from_tokens_to_sentence(predictions[i], stop_end))
            except tf.errors.OutOfRangeError:
                break

    def _from_tokens_to_sentence(self, tokens, stop_end):
        return (self._model_builder.get_tokenization_handler().
                from_tokens_to_sentence(tokens, stop_end))
