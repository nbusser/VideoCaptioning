"""Module used to extract video frames from a video"""

import cv2
import numpy as np

def count_video_frames(video_url):
    """Returns the number of frames of the video"""
    video = cv2.VideoCapture(video_url)
    return int(np.round(video.get(cv2.CAP_PROP_FRAME_COUNT)))

def inceptionv3_frame_preprocess(frame):
    """Turns the video into inceptionv3 format

    -Resizes the video to 299*299
    -Converts the values from [0;255] int representation to [-1;1] float representation
    """

    frame = cv2.resize(frame, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)

    frame = frame.astype(np.float32)
    frame = frame/255.0
    frame = (frame*2.0)-1.0

    return frame

def read_video(video_path, preprocess_inceptionv3=False):
    """This function extracts the frames of the video located in video_path

    Keyword arguments:
    video_path -- path of the video
    preprocess_inceptionv3 -- if set, the video will be formated to match inceptionv3

    Returns:
    A numpy array shaped [number_of_frames, width, height, number_of_channels]
    If preprocess_inceptionv3 is set, width and height should be equal to 299
    """

    video = cv2.VideoCapture(video_path)
    frames = []

    while video.isOpened():
        rval, frame = video.read()
        if rval:
            # OpenCV uses BRG color respresentation instead of RGB.
            # Turnes BRG to RGB representation
            frame = frame[..., ::-1]

            # Doing preprocessing if wanted
            if preprocess_inceptionv3:
                frame = inceptionv3_frame_preprocess(frame)

            frames.append(frame)
        else:
            break

    video.release()
    return np.asarray(frames)
