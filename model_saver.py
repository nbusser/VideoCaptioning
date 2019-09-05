"""This module is used to save and restore models using json and checkpoints files"""

import datetime
import json
import os
from model_builder import ModelBuilder
from model_trainer import ModelTrainer
from model_predictor import ModelPredictor

def build_save_filename(prefix, n_videos, n_captions_per_video, batch_size):
    """This tool builds a name from a prefix, the model parameters and the current date"""
    filename = (prefix + '_' + str(n_videos) + '_' + str(n_captions_per_video)
                + '_' + str(batch_size) + '_')

    date = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
    filename = filename + date

    return filename

def restore_model(checkpoint_filename, video_retriever_generator,
                  selector, extractor):
    """This function restores a model from a tf checkpoint

    Json filename is asserted to be the same as checpoint filename, but with json file extension

    -Recovers the model's parameters via the json file
    -Builds the model using this parameters
    -Prepare the tensorflow graph and get neural network operations
    """

    json_filename = checkpoint_filename + '.json'
    with open(json_filename, 'r', encoding='utf-8') as json_file:
        params = json.load(json_file)

    builder = ModelBuilder(params["training_videos_names"], params["testing_videos_names"],
                           params["n_captions_per_video"], params["feature_n_frames"],
                           video_retriever_generator, selector, extractor)
    model = params["model"]
    builder.create_model(model["enc_units"], model["dec_units"], model["rnn_layers"],
                         model["embedding_dims"], model["learning_rate"],
                         model["dropout_rate"], model["bi_encoder"])

    builder.prepare_training(params["batch_size"])

    model_saver = ModelSaver(os.path.dirname(checkpoint_filename),
                             os.path.basename(checkpoint_filename))

    return builder, model_saver, params

def model_trainer_from_checkpoint(checkpoint_filename, video_retriever_generator,
                                  selector, extractor):
    """This function restores a model from a tf checkpoint and creates a ModelTrainer"""
    builder, model_saver, params = restore_model(checkpoint_filename, video_retriever_generator,
                                                 selector, extractor)

    model_trainer = ModelTrainer(model_saver, builder, params["epoch"], float(params["best_loss"]))

    model_trainer.load_last_checkpoint()

    return model_trainer

def model_predictor_from_checkpoint(checkpoint_filename, videos_retriever_generator,
                                    selector, extractor):
    """This function restores a model from a tf checkpoint and creates a ModelPredictor"""
    builder, model_saver, _ = restore_model(checkpoint_filename, videos_retriever_generator,
                                            selector, extractor)

    model_predictor = ModelPredictor(model_saver, builder)

    model_predictor.load_last_checkpoint()

    return model_predictor


class ModelSaver(object):
    """This classe manages the outputs (saving) of the video captioning model"""
    def __init__(self, checkpoint_folder, checkpoint_filename):
        self._checkpoint_folder = checkpoint_folder

        self._savefile_name = os.path.join(self._checkpoint_folder, checkpoint_filename)
        self._savefile_json = self._savefile_name + '.json'

    @staticmethod
    def from_generated_filename(model_builder, checkpoint_folder, prefix=''):
        """Creates a ModelSaver using the build_save_filename function"""
        savefile_name = build_save_filename(prefix,
                                            model_builder.get_n_videos(),
                                            model_builder.get_n_captions_per_video(),
                                            model_builder.get_batch_size())

        return ModelSaver(checkpoint_folder, savefile_name)

    def save_model_json(self, model_builder, epoch, best_loss):
        """Save all the model parameters into a json file"""
        json_dict = model_builder.model_to_dict()
        json_dict['epoch'] = epoch
        json_dict['best_loss'] = str(best_loss)

        with open(self._savefile_json, 'w', encoding='utf-8') as json_file:
            json.dump(json_dict, json_file, ensure_ascii=False, indent=4)

    def get_savefile_name(self):
        """Returns the name of the savefile"""
        return self._savefile_name
