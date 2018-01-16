import numpy as np
import cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt


def slide_window(
        img,
        x_start_stop=[None, None],
        y_start_stop=[None, None],
        xy_window=(64, 64),
        xy_overlap=(0.5, 0.5)):
    """
    Calculates a list of windows given an image, bounding box, window size and overlap fraction
    :param img:
    :param x_start_stop:
    :param y_start_stop:
    :param xy_window:
    :param xy_overlap:
    :return:
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draw bounding boxes on an images
    :param img:
    :param bboxes:
    :param color:
    :param thick:
    :return:
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


if __name__ == "__main__":
    ystart = 380
    ystop = 550
    scale = 1.0
    bbox_list.append(find_cars(img, ystart, ystop, scale, svc, X_scaler,
                               orient, pix_per_cell, cell_per_block,
                               spatial_size, hist_bins))

    ystart = 400
    ystop = 600
    scale = 1.5
    bbox_list.append(find_cars(img, ystart, ystop, scale, svc, X_scaler,
                               orient, pix_per_cell, cell_per_block,
                               spatial_size, hist_bins))

    ystart = 400
    ystop = 656
    scale = 2.0
    bbox_list.append(find_cars(img, ystart, ystop, scale, svc, X_scaler,
                               orient, pix_per_cell, cell_per_block,
                               spatial_size, hist_bins))

    image1 = imread('test_images/test3.jpg')
    windows1 = slide_window(image1, x_start_stop=(320, 1024), y_start_stop=(370, 562),
                           xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    window_img1 = draw_boxes(image1, windows1, color=(0, 0, 255), thick=6)

    plt.figure()
    plt.imshow(window_img1)

    image2 = imread('test_images/test1.jpg')
    windows2 = slide_window(image2, x_start_stop=(280, 1048), y_start_stop=(400, 592),
                            xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    window_img2 = draw_boxes(image2, windows2, color=(0, 0, 255), thick=6)

    plt.figure()
    plt.imshow(window_img2)

    image3 = imread('test_images/test5.jpg')
    windows3 = slide_window(image2, x_start_stop=(256, 1280), y_start_stop=(428, 620),
                            xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    window_img3 = draw_boxes(image3, windows3, color=(0, 0, 255), thick=6)

    plt.figure()
    plt.imshow(window_img3)

    plt.show(block=True)