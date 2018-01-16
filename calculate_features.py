import numpy as np
import cv2
from skimage.feature import hog
from matplotlib.image import imread


class HogParameters:
    def __init__(self, orientations, pixels_per_cell, cells_per_block)
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block


def color_hist(img, nbins=32, bins_range=(0, 1)):
    """
    Compute color histogram features
    :param img:
    :param nbins:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the colour channels separately
    hist1 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    hist2 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    hist3 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = hist1[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist1, hist2, hist3, bin_centers, hist_features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """
    Compute color histogram features
    :param img:
    :param color_space:
    :param size:
    :return:
    """
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# Define a function to return HOG features and visualization
def get_hog_features(
        img,
        hog_parameters = HogParameters(9,8,2),
        vis=False,
        feature_vec=True):
    """
    Calculate HOG features
    :param img:
    :param hog_parameters:
    :param vis:
    :param feature_vec:
    :return:
    """
    if vis:
        features, hog_image = hog(img,
                                  orientations=hog_parameters.orientations,
                                  pixels_per_cell=(hog_parameters.pixels_per_cell, hog_parameters.pixels_per_cell),
                                  cells_per_block=(hog_parameters.cells_per_block, hog_parameters.cells_per_block),
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=False)
        return features, hog_image
    else:
        features = hog(img,
                       orientations=hog_parameters.orientations,
                       pixels_per_cell=(hog_parameters.pixels_per_cell, hog_parameters.pixels_per_cell),
                       cells_per_block=(hog_parameters.cells_per_block, hog_parameters.cells_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vec)
        return features


def extract_hog_features(imgs,
                         colour_space='RGB',
                         hog_parameters = HogParameters(9,8,2),
                         hog_channel=0):
    """
    Extract HOG features from a list of images
    :param imgs:
    :param colour_space:
    :param hog_parameters:
    :param hog_channel:
    :return:
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = imread(file)
        # apply color conversion if other than 'RGB'
        if colour_space != 'RGB':
            if colour_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif colour_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif colour_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif colour_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif colour_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(
                    get_hog_features(
                        feature_image[:,:,channel],
                        hog_parameters=hog_parameters,
                        vis=False,
                        feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(
                feature_image[:,:,hog_channel],
                hog_parameters=hog_parameters,
                vis=False,
                feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


def single_img_features(
        img,
        color_space='RGB',
        spatial_size=(32, 32),
        hist_bins=32,
        hog_parameters=HogParameters(9,8,2),
        hog_channel=0,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(
                    get_hog_features(feature_image[:,:,channel],
                                     hog_parameters=hog_parameters,
                                     vis=False,
                                     feature_vec=True))
        else:
            hog_features = get_hog_features(
                feature_image[:,:,hog_channel],
                hog_parameters=hog_parameters,
                vis=False,
                feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


if __name__ == "__main__":
    import load_samples
    import matplotlib.pyplot as plt
    from random import shuffle
    kitti = load_samples.get_kitti_car_images()
    shuffle(kitti)
    image = imread(kitti[0])

    # Color histogram
    fig, axes = plt.subplots(1, 4)
    hist1, hist2, hist3, bin_centers, hist_features = color_hist(image)
    axes[0].imshow(image)
    axes[0].set_title('Example Car Image')
    axes[1].set_title('R Channel Histogram')
    axes[1].bar(bin_centers, hist1[0])
    axes[1].set_xlim(0.0, 1.0)
    axes[2].set_title('G Channel Histogram')
    axes[2].bar(bin_centers, hist2[0])
    axes[2].set_xlim(0.0, 1.0)
    axes[3].set_title('B Channel Histogram')
    axes[3].bar(bin_centers, hist3[0])
    axes[3].set_xlim(0.0,1.0)

    # Spatial binning of color
    feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))
    fig = plt.figure()
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')

    # HOG
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    features, hog_image = get_hog_features(
        gray_img,
        hog_parameters=HogParameters(orientations=orient, pixels_per_cell=8, cells_per_block=cell_per_block),
        vis=True,
        feature_vec=True)
    fig,axes = plt.subplots(1,2)
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Example Car Image')
    axes[1].imshow(hog_image, cmap='gray')
    axes[1].set_title('HOG Visualization')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show(block=True)