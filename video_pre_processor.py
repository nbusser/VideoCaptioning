"""This module introduces VideoPreProcessor, a tool to preprocess, save and
retrieve video features"""

import os
import json
import numpy as np

class VideoPreProcessor(object):
    """This class is used to preprocess videos by category and to retrieve saved features

    Here is the applied politic:
    -Videos are saved by category
    -For each category is associated a .npz numpy archive containing all the video
    features of its videos
    -Since features are heavy in RAM, you can only retrieve the videos with a generator
    """

    def __init__(self, selector, extractor, feature_path):
        """Initializer

        To create a VideoPreProcessor, you need to give a MsrVttSelector and
        a MsrVttExtractor.
        They are both used to retrieve data from the MSR-VTT dataset.
        """

        self._selector = selector
        self._extractor = extractor
        self._feature_path = feature_path
        self._json_n_frames_path = os.path.join(self._feature_path, 'n_frames.json')

    def pre_process_category(self, category, n_frames, force=False):
        """Creates a numpy archive for a specific category

        Retrieves the names of videos in the current category
        Extracts the features using inceptionv3. Uses the n_frames determined higher
        Saves the results in a numpy archive ([video_numpy_features_folder]/CAT_NAME.npz)
        Saves the n_frames in the json file containing n_frames of all categories

        If force argument is False, don't pre-process the category if the file already exists
        """

        saved_feature_path = self._get_category_path(category)
        filename = self._get_feature_file_path(category)

        if os.path.exists(filename) and not force:
            print("File {} already exists. If you want to overwrite, use force parameter"
                  .format(filename))
            return

        print("Processing {} category".format(category.name))

        videos_names = self._selector.select_videos(categories=[category])

        bottlenecks = self._extractor.inceptionv3_preprocess_videos(videos_names, n_frames,
                                                                    verbose=1)

        np.savez(saved_feature_path,
                 **{name:value for name, value in zip(videos_names, bottlenecks)})

        self._save_json_n_frames(category, n_frames)

        print("File {}.npz created".format(category.name))

    def pre_process_categories(self, categories, n_frames, force=False):
        """Iterates over categories and preprocesses it"""
        if not isinstance(categories, list):
            categories = [categories]

        for category in categories:
            self.pre_process_category(category, n_frames, force)

    def _get_category_path(self, category):
        return os.path.join(self._feature_path, category.name)

    def _get_feature_file_path(self, category):
        return self._get_category_path(category) + '.npz'

    def video_retriever_generator(self, videos_names):
        """This generator is used to retrieve previously saved video features

        - Creates a dict associating category value with opened npz file
        - Yields the features associated to videos_names by picking in the right npz file
        - Closes all the categories npz files
        """

        categories = self._selector.get_categories(videos_names)

        categories_npz = {}
        for category in set(categories):
            categories_npz[category.value] = np.load(self._get_feature_file_path(category))

        for video_name, category in zip(videos_names, categories):
            yield categories_npz[category.value][video_name]

        for category in set(categories):
            categories_npz[category.value].close()

    def get_n_frames(self, videos_names):
        """Returns the number of frames of each videos contained in videos_names

        To perform this, uses the json file containing n_frames of all preprocessed
        categories"""

        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        categories = self._selector.get_categories(videos_names)

        try:
            n_frames_json = self._get_json_n_frames()
        except FileNotFoundError:
            raise FileNotFoundError(
                "n_frames.json missing. Please preprocess your videos again")

        n_frames = []
        try:
            for category in categories:
                n_frames.append(n_frames_json[category.name])
        except KeyError:
            raise KeyError("Category {} has not been preprocessed yet", category.name)

        return n_frames

    def get_uniform_n_frames(self, videos_names):
        """Returns, if it exists, the unique number of frame of all videos in videos_names

        If videos in videos_names belongs to multiple categories with heterogen
        number of frames, returns None
        """

        n_frames = self.get_n_frames(videos_names)

        if n_frames is None:
            return None

        if not all(x == n_frames[0] for x in n_frames):
            print("No uniform n_frames found")
            return None

        return n_frames[0]

    def _save_json_n_frames(self, category, n_frames):
        """Save the n_frame of the preprocessed category to the json file"""
        n_frames_cat = self._get_json_n_frames()
        n_frames_cat[category.name] = n_frames

        with open(self._json_n_frames_path, 'w', encoding='utf-8') as json_file:
            json.dump(n_frames_cat, json_file)


    def _get_json_n_frames(self):
        """Retrieves json file info"""
        with open(self._json_n_frames_path) as json_file:
            n_frames_json = json.load(json_file)
        return n_frames_json
