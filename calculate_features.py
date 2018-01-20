import numpy as np
import cv2
from skimage.feature import hog
from matplotlib.image import imread


class HogParameters:
    def __init__(self, orientations, pixels_per_cell, cells_per_block):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block


def color_hist(img, nbins=32, bins_range=(0, 255)):
    """
    Compute color histogram features
    :param img: The image to compute the colour histogram of
    :param nbins: The number of bins in the colour histogram
    :param bins_range: The range of the bins
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


def bin_spatial(img, color_space='RGB', size=(16, 16)):
    """
    Compute spatial binning of colours
    :param img: The image to compute the spatial colour bins of
    :param color_space: The chosen color space
    :param size: The size to which the image will be resized to construct a color histogram
    :return: A list of spatial color features
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
    :param img: The image to apply HOG feature extraction on
    :param hog_parameters: HogParameters instance
    :param vis: Visualize the HOG gradients or not
    :param feature_vec: Return the features as a 1D vector or not
    :return: All the hog features. If vis is True, then the second (optional) return value is an image with the HOG
             gradients visualized
    """
    if vis:
        features, hog_image = hog(img,
                                  orientations=hog_parameters.orientations,
                                  pixels_per_cell=(hog_parameters.pixels_per_cell, hog_parameters.pixels_per_cell),
                                  cells_per_block=(hog_parameters.cells_per_block, hog_parameters.cells_per_block),
                                  block_norm="L2-Hys",
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=False)
        return features, hog_image
    else:
        features = hog(img,
                       orientations=hog_parameters.orientations,
                       pixels_per_cell=(hog_parameters.pixels_per_cell, hog_parameters.pixels_per_cell),
                       cells_per_block=(hog_parameters.cells_per_block, hog_parameters.cells_per_block),
                       block_norm="L2-Hys",
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vec)
        return features


def extract_features(imgs,
                     colour_space='RGB',
                     spatial_size=(16, 16),
                     hist_bins=32,
                     hog_parameters = HogParameters(9,8,2),
                     hog_channel='ALL',
                     spatial_feat=True, hist_feat=True, hog_feat=True):
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
        file_features = []
        # Read in each one by one
        image = imread(file)
        image = image.astype(np.float32) * 255
        image = image.astype(np.uint8)
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

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            _, _, _, _, hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
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
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


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
    feature_vec = bin_spatial(image, color_space='RGB', size=(16, 16))
    fig = plt.figure()
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')

    # HOG
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features, hog_image1 = get_hog_features(
        hsv_img[:, :, 0],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    features, hog_image2 = get_hog_features(
        hsv_img[:, :, 1],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    features, hog_image3 = get_hog_features(
        hsv_img[:, :, 2],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    hls_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    features, hog_image4 = get_hog_features(
        hls_img[:, :, 0],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    features, hog_image5 = get_hog_features(
        hls_img[:, :, 1],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    features, hog_image6 = get_hog_features(
        hls_img[:, :, 2],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    features, hog_image7 = get_hog_features(
        lab_img[:, :, 0],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    features, hog_image8 = get_hog_features(
        lab_img[:, :, 1],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    features, hog_image9 = get_hog_features(
        lab_img[:, :, 2],
        hog_parameters=HogParameters(orientations=18, pixels_per_cell=8, cells_per_block=2),
        vis=True,
        feature_vec=True)
    fig,axes = plt.subplots(4,3)
    for ax in axes.flatten():
        ax.axis('off')
    axes[0][0].imshow(image, cmap='gray')
    axes[0][0].set_title('Example Car Image')
    axes[1][0].imshow(hog_image1, cmap='gray')
    axes[1][0].set_title('HOG Visualization H')
    axes[1][1].imshow(hog_image2, cmap='gray')
    axes[1][1].set_title('HOG Visualization S')
    axes[1][2].imshow(hog_image3, cmap='gray')
    axes[1][2].set_title('HOG Visualization V')
    axes[2][0].imshow(hog_image4, cmap='gray')
    axes[2][0].set_title('HOG Visualization H')
    axes[2][1].imshow(hog_image5, cmap='gray')
    axes[2][1].set_title('HOG Visualization L')
    axes[2][2].imshow(hog_image6, cmap='gray')
    axes[2][2].set_title('HOG Visualization S')
    axes[3][0].imshow(hog_image7, cmap='gray')
    axes[3][0].set_title('HOG Visualization L')
    axes[3][1].imshow(hog_image8, cmap='gray')
    axes[3][1].set_title('HOG Visualization A')
    axes[3][2].imshow(hog_image9, cmap='gray')
    axes[3][2].set_title('HOG Visualization B')



    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show(block=True)