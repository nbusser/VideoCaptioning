"""Extractor adapted to MSR-VTT dataset"""

from video_zip_extractor import VideoZipExtractor

def from_video_name_to_zip_url(video_name):
    """Simple string conversion from video name to video name in archive"""
    return 'train-video/' + video_name + '.mp4'

def name_preprocess(videos_names):
    """Name conversion for several names"""
    if not isinstance(videos_names, list):
        videos_names = [videos_names]

    return [from_video_name_to_zip_url(name) for name in videos_names]

class MsrVttExtractor(VideoZipExtractor):
    """MsrVtt zip extractor class

    This class has the same behavior than VideoZipExtractor but uses video_names
    instead of zip urls, which is more compatible with MsrVttLoader class"""

    def inceptionv3_preprocess_videos(self, videos_names, n_frames, verbose=0):
        """Inceptionv3 video preprocessing using VideoZipExtractor"""

        videos_zip_urls = name_preprocess(videos_names)

        return super(MsrVttExtractor, self).inceptionv3_preprocess_videos(videos_zip_urls,
                                                                          n_frames, verbose=verbose)

    def get_videos(self, videos_names):
        """Video frames getter using VideoZipExtractor module"""
        videos_zip_urls = name_preprocess(videos_names)
        return super(MsrVttExtractor, self).get_videos(videos_zip_urls)

    def get_n_frames(self, videos_names):
        """Returns an array with the lengths of the specified videos"""
        videos_zip_urls = name_preprocess(videos_names)
        return super(MsrVttExtractor, self).get_n_frames(videos_zip_urls)
