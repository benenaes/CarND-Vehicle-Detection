import cv2
import numpy as np
import glob
from matplotlib.image import imread
from calculate_features import single_img_features, get_hog_features, HogParameters


class SlidingWindowAreaDefinition:
    def __init__(self, x_start, x_stop, y_start, y_stop, scale):
        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop
        self.scale = scale


def search_windows(
        image, windows, clf, scaler, colour_space='RGB',
        spatial_size=(32, 32), hist_bins=32,
        # hist_range=(0, 256),
        hog_parameters=HogParameters(9,8,2),
        hog_channel=0, spatial_feat=False,
        hist_feat=False, hog_feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(
            test_img,
            color_space=colour_space,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            hog_parameters=hog_parameters,
            hog_channel=hog_channel,
            spatial_feat=spatial_feat,
            hist_feat=hist_feat,
            hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(
        img,
        window_area_def=SlidingWindowAreaDefinition(0,1280,0,720,1),
        clf,
        X_scaler,
        hog_parameters=HogParameters(18,8,2),
        # spatial_size,
        # hist_bins
        ):
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_to_search = img[
                    window_area_def.y_start:window_area_def.y_stop,
                    window_area_def.x_start:window_area_def.y_start,
                    :]
    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    # if color_space != 'RGB':
    #     if color_space == 'HSV':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     elif color_space == 'LUV':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    #     elif color_space == 'HLS':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #     elif color_space == 'YUV':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #     elif color_space == 'YCrCb':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # else:
    #     feature_image = np.copy(img)

    if window_area_def.scale != 1:
        imshape = img_to_search.shape
        img_to_search = cv2.resize(
            img_to_search,
            (np.int(imshape[1] / window_area_def.scale), np.int(imshape[0] / window_area_def.scale)))

    ch1 = img_to_search[:, :, 0]
    ch2 = img_to_search[:, :, 1]
    ch3 = img_to_search[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // hog_parameters.pixels_per_cell) - hog_parameters.cells_per_block + 1
    nyblocks = (ch1.shape[0] // hog_parameters.pixels_per_cell) - hog_parameters.cells_per_block + 1
    nfeat_per_block = hog_parameters.orientations * hog_parameters.cells_per_block ** 2

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
            spatial_features = [] # bin_spatial(subimg, size=spatial_size)
            hist_features = [] #color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * window_area_def.scale)
                ytop_draw = np.int(ytop * window_area_def.scale)
                win_draw = np.int(window * window_area_def.scale)
                cv2.rectangle(
                    img=draw_img,
                    pt1=(xbox_left + window_area_def.x_start, ytop_draw + window_area_def.y_start),
                    pt2=(xbox_left + window_area_def.x_start + win_draw, ytop_draw + win_draw + window_area_def.y_start),
                    color=(0, 0, 255),
                    thickness=6)

    return draw_img


if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt

    clf = pickle.load("precision-svm.p")
    hog_scaler = pickle.load("hog-scaler.p")
    hog_parameters = HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2)
    test_images = glob.glob("test_images/*.jpg")
    for test_file in test_images:
        img = imread(test_file)
        draw_img = find_cars(
            img=img,
            window_area_def=SlidingWindowAreaDefinition(0,1280,0,720,1),
            clf=clf,
            X_scaler=hog_scaler, hog_parameters=HogParameters(18,8,2))
        plt.figure()
        plt.imshow(draw_img)
    plt.show(block=True)
