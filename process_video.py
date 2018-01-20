from moviepy.editor import VideoFileClip
import pickle

from process_frame import process_frame
from calculate_features import HogParameters


def process_video(input_file, output_file):
    """
    Detect cars within an entire MPEG-4 video
    :param input_file: Original video
    :param output_file: Video where blue bounding boxes are drawn on detected cars
    :return:
    """
    with open('all-features-rbf-svm.p', 'rb') as svm_fd:
        clf = pickle.load(svm_fd)
    with open('all-features-scaler.p', 'rb') as scaler_fd:
        hog_scaler = pickle.load(scaler_fd)
    hog_parameters = HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2)
    clip = VideoFileClip(input_file)
    test_clip = clip.fl_image(
        lambda frame: process_frame(frame, clf=clf, norm_scaler=hog_scaler, hog_parameters=hog_parameters, spatial_size=(16, 16), hist_bins=32))
    test_clip.write_videofile(output_file, audio=False)


if __name__ == "__main__":
    process_video(input_file="project_video.mp4", output_file="project_output.mp4")