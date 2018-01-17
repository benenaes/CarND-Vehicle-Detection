from moviepy.editor import VideoFileClip
import pickle

from process_frame import process_frame
from calculate_features import HogParameters


def process_video(input_file, output_file):
    clf = pickle.load("precision-svm.p")
    hog_scaler = pickle.load("hog-scaler.p")
    hog_parameters = HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2)
    clip = VideoFileClip(input_file)
    test_clip = clip.fl_image(
        lambda frame: process_frame(frame, clf=clf, norm_scaler=hog_scaler, hog_parameters=hog_parameters))
    test_clip.write_videofile(output_file, audio=False)


if __name__ == "__main__":
    # clip = VideoFileClip("project_video.mp4")
    # clip.save_frame("test_images/close_car.jpg", t=29)
    process_video(input_file="project_video.mp4", output_file="project_output.mp4")