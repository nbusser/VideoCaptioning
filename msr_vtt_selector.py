"""This module is an interface to use MSR-VTT dataset"""

import os
import zipfile as zf
import json
from enum import Enum
from collections import defaultdict
import numpy as np
from functools import reduce

class Category(Enum):
    """This enum contains all the MSR-VTT categories"""
    MUSIC = 0
    PEOPLE = 1
    GAMING = 2
    SPORT = 3
    NEWS = 4
    EDUCATION = 5
    TV = 6
    MOVIE = 7
    ANIMATION = 8
    VEHICLES = 9
    HOWTO = 10
    TRAVEL = 11
    SCIENCE = 12
    ANIMALS = 13
    KIDS = 14
    DOCUMENTARY = 15
    FOOD = 16
    COOKING = 17
    BEAUTY = 18
    ADVERTISEMENT = 19

class MsrVttSelector(object):
    """MSR-VTT dataset specific select"""

    def __init__(self, json_zip_file):
        """Inits the loader

        Keyword arguments:
        train_video_zip_file -- url of MSR-VTT zip file containing videos
        json_zip_file -- url of MSR-VTT zip file containing json data
        """

        # MSR-VTT contains 10,000 videos
        self._dataset_size = 10000
        # There is 20 captions per video
        self._n_captions_video = 20

        with zf.ZipFile(json_zip_file, 'r') as myzip:
            json_file = myzip.extract('videodatainfo_2017.json', '/tmp')

        self._json_data = open(json_file)
        self._json_data = json.load(self._json_data)

        os.remove(json_file)

        # Builds a dictionary video_name -> [captions]
        self._caption_dict = defaultdict(list)
        for elem in self._json_data["sentences"]:
            video_id = elem["video_id"]
            self._caption_dict[video_id].append(elem["caption"])

        # This line removes a double ' ' character which can disturb further
        # caption handling
        self._caption_dict["video8128"][0] = " ".join(
            self._caption_dict["video8128"][0].split())

    def select_videos(self, limit=None, categories=None, random=False):
        """Select specific a number of videos from specified categories

        Keyword arguments:
        limit -- max number of video to select. If None, iter over all the dataset
        categories -- list of chosen categories. Selected videos will only belong
                      to this categories. None value means that all categories are chosen
        random -- is set to True, randomize picking order

        Returns:
        A list of video names
        """

        if limit is None:
            limit = np.inf

        if categories is None:
            categories = range(len(Category))
        else:
            if not isinstance(categories, list):
                categories = [categories]

            categories = [cat.value for cat in categories]

        if not random:
            indices = range(self._dataset_size)
        else:
            indices = np.arange(self._dataset_size)
            np.random.shuffle(indices)

        video_names = []

        for i in indices:
            if len(video_names) >= limit:
                break

            if i >= len(self._json_data["videos"]):
                break

            video_infos = self._json_data["videos"][i]
            if video_infos["category"] in categories:
                video_names.append(video_infos["video_id"])

        return video_names

    def get_captions(self, videos_names, format_function=None):
        """For a given set of videos, returns the associated captions"""
        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        captions = [self._caption_dict[video_name] for video_name in videos_names]
        if format_function is not None:
            captions = [[format_function(caption) for caption in caption_list]
                        for caption_list in captions]
        return captions

    def get_n_captions(self, videos_names, n_captions_per_vid, format_function=None):
        """For a given set of videos, returns the n first associated captions"""
        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        captions = [self._caption_dict[video_name][:n_captions_per_vid]
                    for video_name in videos_names]
        if format_function is not None:
            captions = [[format_function(caption) for caption in caption_list]
                        for caption_list in captions]
        return captions

    def get_flattened_captions(self, videos_names, format_function=None):
        """For a given set of videos, returns all the captions in a one dimension list"""
        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        captions = self.get_captions(videos_names)
        captions = reduce(lambda x, y: x+y, captions)
        if format_function is not None:
            captions = list(map(format_function, captions))
        return captions

    def get_flattened_n_captions(self, videos_names, n_captions_per_vid, format_function=None):
        """For a given set of videos, returns n captions per video in a one dimension list"""
        if not isinstance(videos_names, list):
            videos_names = [videos_names]

        flat_captions = self.get_flattened_captions(videos_names)

        flat_captions = [caption for i, caption in enumerate(flat_captions)
                         if i % self._n_captions_video < n_captions_per_vid]
        if format_function is not None:
            flat_captions = list(map(format_function, flat_captions))
        return flat_captions

    def get_categories(self, videos_names):
        """Returns the categories of the videos represented by videos_names"""
        if not isinstance(videos_names, list):
            videos_names = [videos_names]
        return [Category(self._json_data["videos"][int(video_name[5:])]["category"])
                for video_name in videos_names]

def caption_format_add_end(caption):
    """Standard caption preprocessing function"""
    return caption + ' <end>'
