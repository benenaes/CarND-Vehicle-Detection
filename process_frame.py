import numpy as np
from scipy.ndimage.measurements import label
import cv2

from search_windows import find_cars, SlidingWindowAreaDefinition
from heatmap import add_heat, apply_threshold, draw_labeled_bboxes

import collections

heatmap_history = collections.deque(maxlen=6)


def process_frame(frame, clf, norm_scaler, hog_parameters, spatial_size, hist_bins):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    boxes = []

    window_area_def1 = SlidingWindowAreaDefinition(
        x_start=550,
        x_stop=1024,
        y_start=370,
        y_stop=498,
        scaleX=1.0,
        scaleY=1.0
        )

    found_cars = find_cars(hsv_frame,
                           clf=clf,
                           X_scaler=norm_scaler,
                           window_area_def=window_area_def1,
                           hog_parameters=hog_parameters,
                           spatial_size=spatial_size,
                           hist_bins=hist_bins)
    boxes.append(found_cars)

    window_area_def2 = SlidingWindowAreaDefinition(
        x_start=530,
        x_stop=1144,
        y_start=390,
        y_stop=534,
        scaleX=1.5,
        scaleY=1.5
        )
    found_cars = find_cars(hsv_frame,
                           clf=clf,
                           X_scaler=norm_scaler,
                           window_area_def=window_area_def2,
                           hog_parameters=hog_parameters,
                           spatial_size=spatial_size,
                           hist_bins=hist_bins)
    boxes.append(found_cars)

    window_area_def3 = SlidingWindowAreaDefinition(
        x_start=480,
        x_stop=1280,
        y_start=400,
        y_stop=592,
        scaleX=2.0,
        scaleY=2.0
        )
    found_cars = find_cars(hsv_frame,
                           clf=clf,
                           X_scaler=norm_scaler,
                           window_area_def=window_area_def3,
                           hog_parameters=hog_parameters,
                           spatial_size=spatial_size,
                           hist_bins=hist_bins)
    boxes.append(found_cars)

    window_area_def4 = SlidingWindowAreaDefinition(
        x_start=944,
        x_stop=1280,
        y_start=380,
        y_stop=620,
        scaleX=3.0,
        scaleY=2.5
        )
    found_cars = find_cars(hsv_frame,
                           clf=clf,
                           X_scaler=norm_scaler,
                           window_area_def=window_area_def4,
                           hog_parameters=hog_parameters,
                           spatial_size=spatial_size,
                           hist_bins=hist_bins)
    boxes.append(found_cars)

    window_area_def5 = SlidingWindowAreaDefinition(
        x_start=896,
        x_stop=1280,
        y_start=396,
        y_stop=636,
        scaleX=4.0,
        scaleY=3.0
        )
    found_cars = find_cars(hsv_frame,
                           clf=clf,
                           X_scaler=norm_scaler,
                           window_area_def=window_area_def5,
                           hog_parameters=hog_parameters,
                           spatial_size=spatial_size,
                           hist_bins=hist_bins)
    boxes.append(found_cars)

    boxes = [item for sublist in boxes for item in sublist]

    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    heat, single_frame_heat = add_heat(heat, boxes, heatmap_history)
    heatmap_history.append(single_frame_heat)
    heat = apply_threshold(heat, 5.5)

    labels = label(heat)
    result_img = np.copy(frame)
    result_img = draw_labeled_bboxes(result_img, labels)

    return result_img


if __name__ == "__main__":
    import pickle
    import glob
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from calculate_features import HogParameters

    with open('all-features-rbf-svm.p', 'rb') as svm_fd:
        clf = pickle.load(svm_fd)
    with open('all-features-scaler.p', 'rb') as scaler_fd:
        hog_scaler = pickle.load(scaler_fd)
    hog_parameters = HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2)
    test_images = glob.glob("test_images/*.jpg")
    for test_file in test_images:
        img = imread(test_file)
        draw_img = process_frame(
            frame=img,
            clf=clf,
            norm_scaler=hog_scaler,
            hog_parameters=HogParameters(18,8,2),
            spatial_size=(16, 16),
            hist_bins=32)
        plt.figure()
        plt.imshow(draw_img)
        plt.show(block=True)