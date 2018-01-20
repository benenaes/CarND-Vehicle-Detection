import cv2
import numpy as np
import glob
from matplotlib.image import imread
import matplotlib.pyplot as plt
from calculate_features import get_hog_features, HogParameters, bin_spatial, color_hist


class SlidingWindowAreaDefinition:
    def __init__(self, x_start, x_stop, y_start, y_stop, scaleX, scaleY):
        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop
        self.scaleX = scaleX
        self.scaleY = scaleY


class BoundingBox:
    def __init__(self, box, probability):
        self.box = box
        self.probability = probability


def find_cars(img,
              clf,
              feature_scaler,
              window_area_def=SlidingWindowAreaDefinition(0,1280,0,720,1,1),
              hog_parameters=HogParameters(18,8,2),
              spatial_size=(16, 16),
              hist_bins=32,
              visualize_candidate_windows=False):
    """
    Extract the feature vector for all the sliding windows within an image and predict which sliding window is a
    candidate car sliding window.
    :param img: Original image
    :param clf: A classifier that calculates the probability for each sliding window to be a car or not a car
    :param feature_scaler: A normalization scaler for each feature vector of a sliding window
    :param window_area_def: The definition of the area where sliding windows are applied (SlidingWindowAreaDefinition instance)
    :param hog_parameters: HOG parameters to be used
    :param spatial_size: The parameters for the spatial colour binning (a pair containing the sample size in each dimension)
    :param hist_bins: Number of colour histogram bins
    :param visualize_candidate_windows: Visualize all the sliding windows within an image that are candidate cars
    :return: A list of BoundingBox instances corresponding to all candidate car sliding windows
    """
    boxes = []

    img_to_search = img[
                    window_area_def.y_start:window_area_def.y_stop,
                    window_area_def.x_start:window_area_def.x_stop,
                    :]

    if window_area_def.scaleX != 1:
        imshape = img_to_search.shape
        img_to_search = cv2.resize(
            img_to_search,
            (np.int(imshape[1] / window_area_def.scaleX), np.int(imshape[0] / window_area_def.scaleY)))

    ch1 = img_to_search[:, :, 0]
    ch2 = img_to_search[:, :, 1]
    ch3 = img_to_search[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // hog_parameters.pixels_per_cell) - hog_parameters.cells_per_block + 1
    nyblocks = (ch1.shape[0] // hog_parameters.pixels_per_cell) - hog_parameters.cells_per_block + 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // hog_parameters.pixels_per_cell) - hog_parameters.cells_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, hog_parameters, feature_vec=False)
    hog2 = get_hog_features(ch2, hog_parameters, feature_vec=False)
    hog3 = get_hog_features(ch3, hog_parameters, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * hog_parameters.pixels_per_cell
            ytop = ypos * hog_parameters.pixels_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_to_search[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            _, _, _, _, hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = feature_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = clf.predict_proba(test_features)

            if test_prediction[0][1] > 0.7:
                xbox_left = np.int(xleft * window_area_def.scaleX)
                ytop_draw = np.int(ytop * window_area_def.scaleY)
                win_draw_x = np.int(window * window_area_def.scaleX)
                win_draw_y = np.int(window * window_area_def.scaleY)
                top_left = (xbox_left + window_area_def.x_start, ytop_draw + window_area_def.y_start)
                bottom_right = (xbox_left + window_area_def.x_start + win_draw_x,
                                ytop_draw + win_draw_y + window_area_def.y_start)
                if visualize_candidate_windows:
                    plt.figure()
                    img_patch = cv2.cvtColor(img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0],:], cv2.COLOR_HSV2RGB)
                    plt.imshow(img_patch)
                    plt.title("Probability: %f" % test_prediction[0][1])

                boxes.append(BoundingBox(box=(top_left, bottom_right), probability=test_prediction[0][1]))

    return boxes


if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt

    with open('all-features-rbf-svm.p', 'rb') as svm_fd:
        clf = pickle.load(svm_fd)
    with open('all-features-scaler.p', 'rb') as scaler_fd:
        feature_scaler = pickle.load(scaler_fd)
    hog_parameters = HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2)
    test_images = glob.glob("test_images/*.jpg")
    # test_images = glob.glob('non-vehicles/Extras/*.png')
    for test_file in test_images:
        img = imread(test_file)
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        boxes = find_cars(
            img=hsv_frame,
            clf=clf,
            feature_scaler=feature_scaler,
            hog_parameters=HogParameters(18, 8, 2),
            window_area_def=SlidingWindowAreaDefinition(0,1280,400,720,1,1),
            spatial_size=(16,16),
            hist_bins=32)
        print(test_file)
        plt.show(block=True)
