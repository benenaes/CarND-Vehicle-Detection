import numpy as np
from scipy.ndimage.measurements import label

from search_windows import find_cars, SlidingWindowAreaDefinition
from heatmap import add_heat, apply_threshold, draw_labeled_bboxes


def process_frame(frame, clf, norm_scaler, hog_parameters):
    # orient = 10
    # pix_per_cell = 8
    # cell_per_block = 2
    # spatial_size = (32, 32)
    # hist_bins = 64

    bbox_list = []

    window_area_def1 = SlidingWindowAreaDefinition(
        x_start=400,
        x_stop=1024,
        y_start=380,
        y_stop=550,
        scale=1.0
        )

    found_cars = find_cars(frame,
                           window_area_def1,
                           clf,
                           norm_scaler,
                           hog_parameters)
    bbox_list.append(found_cars)

    window_area_def2 = SlidingWindowAreaDefinition(
        x_start=400,
        x_stop=1024,
        y_start=400,
        y_stop=660,
        scale=1.5
        )
    found_cars = find_cars(frame,
                           window_area_def2,
                           clf,
                           norm_scaler,
                           hog_parameters)
    bbox_list.append(found_cars)

    window_area_def3 = SlidingWindowAreaDefinition(
        x_start=400,
        x_stop=1024,
        y_start=400,
        y_stop=656,
        scale=2.0
        )
    found_cars = find_cars(frame,
                           window_area_def3,
                           clf,
                           norm_scaler,
                           hog_parameters)
    bbox_list.append(found_cars)

    bbox_list = [item for sublist in bbox_list for item in sublist]

    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    heat = add_heat(heat, bbox_list)
    heat = apply_threshold(heat, 2)

    labels = label(heat)
    result_img = np.copy(frame)
    result_img = draw_labeled_bboxes(result_img, labels)

    return result_img