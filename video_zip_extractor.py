"""This module is used to preprocess videos contained in a zip file"""

import os
import zipfile as zf
import numpy as np
import tensorflow as tf
import video_extractor_utils as utils

def get_pool3_model(inceptionv3_model):
    """Extracts the pool3:0 layer from inceptionv3 model:

    We need to reach the output shape of (n_frames, 2048) by getting the pool3:0
    layer.
    However, Keras' inceptionv3 last hidden layer isn't the pool3:0 layer
    but is the mixed10:0 layer, which outputs has shape (nb_frame, 8, 8, 2048).

    To recover the same dimension as in pool3:0 layer, we need to compute the mean
    in the axis 1 and 2.
    """

    def reduce_mean(hidden):
        """Transforms tf.reduce_mean into a keras layer

        We use a little trick with keras Lambda layer, so, keras accepts to
        use a tensorflow function as output.
        """

        pool3_layer = tf.reduce_mean(hidden, axis=(1, 2))
        return pool3_layer

    inputs = inceptionv3_model.input
    hidden = inceptionv3_model.layers[-1].output
    pool3_layer = tf.keras.layers.Lambda(reduce_mean)(hidden)

    return tf.keras.Model(inputs, pool3_layer)


class VideoZipExtractor(object):
    """Zipped videos preprocessing loader:

    This class can handle videos contained in an archive and preprocess it
    using an inceptionv3 model
    """

    def __init__(self, video_archive_path, inception_model_path=None):
        """Inits the extractor:

        Keyword arguments:
        video_archive_path -- ZIP archive containing the videos
        inception_model_path -- path to a saved keras inceptionv3 model
                                If set to None, no model will be build until
                                calling download_inceptionv3_model method
        """

        self._video_path_zip = video_archive_path

        self._image_feature_extract_model = None

        if inception_model_path is not None:
            self._image_feature_extract_model = get_pool3_model(
                tf.keras.models.load_model(inception_model_path, compile=False))

    def get_inceptionv3_model(self):
        """Inceptionv3 model getter"""
        return self._image_feature_extract_model

    def download_inceptionv3_model(self, save_file=None):
        """Downloads the inceptionv3 model and save it in save_file if desired

        Keyword argument:
        save_file -- if specified, path to save the inceptionv3 model
        """

        inceptionv3_model = tf.keras.applications.InceptionV3(
            include_top=False, weights='imagenet')

        if save_file is not None:
            inceptionv3_model.save(save_file)

        self._image_feature_extract_model = get_pool3_model(inceptionv3_model)

    def _extract_video(self, video_name, preprocess_inceptionv3=False):
        """Unzips a video, then extracts its frames using read_video"""
        with zf.ZipFile(self._video_path_zip, 'r') as myzip:
            video_url = myzip.extract(video_name)
            video = utils.read_video(video_url,
                                     preprocess_inceptionv3=preprocess_inceptionv3)
            os.remove(video_url)

        return video


    def _video_extraction_generator(self, videos_names, n_frames):
        """Generator yielding prepared videos:

        Provides a generator to use model.predict_generator
        Extracts each videos in videos_names and pads/crops it to match n_frames
        """

        for name in videos_names:
            video = self._extract_video(name, preprocess_inceptionv3=True)

            video_n_frames = video.shape[0]

            if video_n_frames > n_frames:
                start_crop_index = np.random.randint(0, video_n_frames-n_frames)
                video = video[start_crop_index:start_crop_index+n_frames][:][:]
            else:
                video = np.pad(video, [(n_frames-video_n_frames, 0), (0, 0),
                                       (0, 0), (0, 0)], mode='constant')
            yield video


    def inceptionv3_preprocess_videos(self, videos_names, n_frames, verbose=0):
        """Computes inceptionv3 bottlenecks:

        Uses inceptionv3 model to preprocess every video in videos_names
        Caution: this function can be very resource-consuming
        Please be sure that yout RAM can handle reshaped_bottlenecks

        Keyword arguments:
        videos_names -- list of video paths regarding the video_archive_path file
                        given in the constructor
        n_frames -- desired number of frames for each videos. All videos will
                    be shaped [n_frames, 299, 299, 3] at the end of preprocessing
        verbose -- sets model_predict level of verbosity

        Returns:
        A numpy array shaped [len(videos_names), n_frames, 2048] corresponding
        to the features of video_names
        """

        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        if self._image_feature_extract_model is None:
            raise Exception("No loaded InceptionV3 model found. " +
                            "Please use method 'download_inceptionv3_model' " +
                            "before runing 'preprocess_videos' method")
         
       
        bottlenecks = self._image_feature_extract_model.predict_generator(
            self._video_extraction_generator(videos_names, n_frames),
            verbose=verbose, steps=len(videos_names), use_multiprocessing=False)

        reshaped_bottlenecks = np.reshape(bottlenecks, (-1, n_frames, 2048))

        return reshaped_bottlenecks

    def get_videos(self, videos_names):
        """Retrieves the videos in videos_names, without any image preprocessing

        Keyword arguments:
        videos_names -- list of video paths regarding the video_archive_path file
                        given in the constructor

        Returns:
        List of videos of native size and with native number of frames"""

        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        videos = []

        for name in videos_names:
            video = self._extract_video(name, preprocess_inceptionv3=False)
            videos.append(video)

        return videos

    def get_n_frames(self, videos_names):
        """Returns a list containing the number of frames of each videos"""

        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        n_frames = []
        with zf.ZipFile(self._video_path_zip, 'r') as myzip:
            for name in videos_names:
                video_url = myzip.extract(name)
                n_frames.append(utils.count_video_frames(video_url))
                os.remove(video_url)

        return n_frames
