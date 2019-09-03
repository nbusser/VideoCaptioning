"""This module is used for its class ModelBuilder that creates and prepare a model"""

import numpy as np
import tensorflow as tf
from msr_vtt_extractor import MsrVttExtractor
from msr_vtt_selector import MsrVttSelector, Category, caption_format_add_end
from seq2seq_model import VideoCaptioningModel
from tokenization_handler import TokenizationHandler

class ModelBuilder(object):
    """This class creates and prepare tensorflow datasets and model operations"""

    def __init__(self, train_videos_names, test_videos_names,
                 n_captions_per_video, n_frames, video_retriever_generator,
                 msr_vtt_selector, msr_vtt_extractor):
        """Inits the builder

        Keyword arguments:
        train_videos_names -- list of names of training videos
        test_videos_names -- list of names of testing videos
        n_captions_per_video -- chosen number of caption per video
        n_frames -- number of frames of video features
        video_retriever_generator -- generator giving access to the video features
        msr_vtt_selector -- selector giving access to the captions
        msr_vtt_extractor -- extractor used to count number of frames
        """

        if not isinstance(train_videos_names, list):
            train_videos_names = [train_videos_names]
        if not isinstance(test_videos_names, list):
            test_videos_names = [test_videos_names]

        self._train_videos_names = train_videos_names
        self._test_videos_names = test_videos_names

        self._n_captions_video = n_captions_per_video

        self._selector = msr_vtt_selector
        self._extractor = msr_vtt_extractor

        videos_names = self._train_videos_names + self._test_videos_names

        captions = self._selector.get_flattened_n_captions(
            videos_names, self._n_captions_video, caption_format_add_end)

        self._token_handler = TokenizationHandler(captions)

        self._feature_n_frames = n_frames

        self._video_retriever_generator = video_retriever_generator

        self._batch_size = None

        self._model = None

        # Model inputs
        self._names = None
        self._features = None
        self._videos_lengths = None
        self._captions = None
        self._captions_lengths = None

        # Data init operations
        self._train_init_op = None
        self._test_init_op = None

        # Model operations
        self._training_logits = None
        self._inference_predictions = None
        self._training_op = None
        self._loss = None
        self._accuracy = None

    def create_model(self, enc_units, dec_units, rnn_layers, embedding_dims,
                     learning_rate, dropout_rate):
        """Creates a VideoCaptioningModel object

        Note that the only difference is vocab_size, that we retrieve from the
        previously created TokenizationHandler
        """
        self._model = VideoCaptioningModel(enc_units, dec_units, rnn_layers,
                                           self._token_handler.get_vocab_size(),
                                           embedding_dims, learning_rate, dropout_rate)


    def prepare_training(self, batch_size, shuffle=True):
        """Creates and prepares the datasets for training by shuffling and batching them

        Asserts: the model has been previously created using create_model

        Builds: model inputs (names, features, video_len, captions, caption_len),
                init operator and model operations (loss, accuracy, training op)
        """
        if self._model is None:
            raise Exception("Model has not been created." +
                            "Please use create_model method before calling this method")

        train_dataset, test_dataset = self._build_datasets()
        next_val, self._train_init_op, self._test_init_op = self._prepare_datasets(
            train_dataset, test_dataset, batch_size, shuffle)

        (self._names, self._features, self._videos_lengths, self._captions,
         self._captions_lengths) = next_val

        (self._training_logits, _, self._inference_predictions, self._training_op,
         self._loss, self._accuracy) = self._build_model_from_next(next_val)

    def get_n_videos(self):
        """Returns number of videos"""
        return len(self._train_videos_names + self._test_videos_names)

    def get_n_captions_per_video(self):
        """Returns the number of selected captions per videos"""
        return self._n_captions_video

    def get_batch_size(self):
        """Returns the batch size"""
        if self._batch_size is None:
            raise Exception("Batch size has not been initialized yet. Please call prepare_training")
        return self._batch_size

    def model_to_dict(self):
        """Returns a dict containing all model parameters"""
        model_infos = {
            'training_videos_names': self._train_videos_names,
            'testing_videos_names': self._test_videos_names,
            'n_captions_per_video': self._n_captions_video,
            'feature_n_frames': self._feature_n_frames,
            'model': self._model.model_parameters_to_dict()
        }

        if self._batch_size is not None:
            model_infos["batch_size"] = self._batch_size

        return model_infos

    def _build_datasets(self):
        train_dataset = build_dataset(self._train_videos_names, self._feature_n_frames,
                                      self._n_captions_video, self._token_handler,
                                      self._selector, self._extractor,
                                      self._video_retriever_generator)

        test_dataset = build_dataset(self._test_videos_names, self._feature_n_frames,
                                     1, self._token_handler,
                                     self._selector, self._extractor,
                                     self._video_retriever_generator)

        return train_dataset, test_dataset

    def _prepare_datasets(self, train_dataset, test_dataset, batch_size, shuffle=True):
        self._batch_size = batch_size

        batch_test = min(len(self._test_videos_names), 700)

        train_dataset = train_dataset.batch(self._batch_size)
        test_dataset = test_dataset.batch(batch_test)

        if shuffle:
            train_dataset = train_dataset.shuffle(len(self._train_videos_names))
            test_dataset = test_dataset.shuffle(len(self._test_videos_names))

        data_iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                                        train_dataset.output_shapes)

        train_init_op = data_iterator.make_initializer(train_dataset)
        test_init_op = data_iterator.make_initializer(test_dataset)

        return data_iterator.get_next(), train_init_op, test_init_op

    def _build_model_from_next(self, next_val):
        _, features, video_len, captions, caption_len = next_val
        return self._model.build_seq2seq_model(features, captions,
                                               video_len, caption_len,
                                               self._token_handler.get_start_token(),
                                               self._token_handler.get_token('<end>'))

    def get_train_init_op(self):
        """Returns train_init_op if it's already created"""
        _verif(self._train_init_op, "train_init_op")
        return self._train_init_op

    def get_test_init_op(self):
        """Returns test_init_op if it's already created"""
        _verif(self._test_init_op, "test_init_op")
        return self._test_init_op

    def get_init_ops(self):
        """Returns train_init_op and test_init_op if it's already created"""
        return self.get_train_init_op(), self.get_test_init_op()

    def get_training_op(self):
        """Returns training_op if it's already created"""
        _verif(self._training_op, "training_op")
        return self._training_op

    def get_loss(self):
        """Returns loss if it's already created"""
        _verif(self._loss, "loss")
        return self._loss

    def get_accuracy(self):
        """Returns accuracy if it's already created"""
        _verif(self._accuracy, "accuracy")
        return self._accuracy

    def get_training_ops(self):
        """Returns the whole operations used in training if it's already created"""
        return self.get_training_op(), self.get_loss(), self.get_accuracy()

    def get_training_logits(self):
        """Returns training_logits if it's already created"""
        _verif(self._training_logits, "training_logits")
        return self._training_logits

    def get_inference_predictions(self):
        """Returns inference_predictions if it's already created"""
        _verif(self._inference_predictions, "inference_predictions")
        return self._inference_predictions

    def get_videos_names(self):
        """Returns videos_names if it's already created"""
        _verif(self._names, "videos_names")
        return self._names

    def get_captions(self):
        """Returns captions if it's already created"""
        _verif(self._captions, "captions")
        return self._captions

    def get_testing_ops(self):
        """Returns the whole operations used in testing if it's already created"""
        return (self.get_videos_names(), self.get_captions(),
                self.get_inference_predictions(), self.get_accuracy())

    def get_tokenization_handler(self):
        """Returns the tokenization_handler of the model"""
        return self._token_handler

def _verif(tensor, operation_name):
    """Simple internal verifier"""
    if tensor is None:
        raise Exception(
            "{} operation has not been created yet. Please build it using prepare_training method".format(operation_name))

def build_dataset(videos_names, features_n_frames, n_captions_per_video,
                  token_handler, selector, extractor, video_retriever_generator):
    """Creates a tf.data.Dataset from simple informations"""
    # Getting the video lengths
    real_video_lengths = extractor.get_n_frames(videos_names)
    # Taking care of video padding during inceptionv3 preprocessing
    video_lengths = [min(length, features_n_frames) for length in real_video_lengths]

    # Retrieving captions
    flattened_captions = selector.get_flattened_n_captions(videos_names,
                                                           n_captions_per_video,
                                                           caption_format_add_end)

    caption_lengths = [len(caption.split(' ')) for caption in flattened_captions]

    # Caption tokenization
    flattened_captions_tokenized = token_handler.tokenize_captions(flattened_captions)

    # Dataset creation
    video_name_dataset = tf.data.Dataset.from_tensor_slices(videos_names)

    features_dataset = tf.data.Dataset.from_generator(
        lambda: map(tuple, video_retriever_generator(videos_names)),
        output_types=tf.float32, output_shapes=(features_n_frames, 2048))

    video_lengths_dataset = tf.data.Dataset.from_tensor_slices(video_lengths)

    caption_dataset = tf.data.Dataset.from_tensor_slices(flattened_captions_tokenized)
    caption_lengths_dataset = tf.data.Dataset.from_tensor_slices(caption_lengths)

    # Repeating n_captions_per_video times
    video_name_dataset = video_name_dataset.flat_map(
        lambda name: tf.data.Dataset.from_tensors(name).repeat(n_captions_per_video))

    features_dataset = features_dataset.flat_map(
        lambda feature: tf.data.Dataset.from_tensors(feature).repeat(n_captions_per_video))

    video_lengths_dataset = video_lengths_dataset.flat_map(
        lambda length: tf.data.Dataset.from_tensors(length).repeat(n_captions_per_video))

    return tf.data.Dataset.zip((video_name_dataset, features_dataset, video_lengths_dataset,
                                caption_dataset, caption_lengths_dataset))
