import numpy as np
import cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt


def slide_window(
        img_width,
        img_height,
        x_start_stop=(None, None),
        y_start_stop=(None, None),
        xy_window=(64, 64),
        xy_overlap=(0.5, 0.5)):
    """
    Calculates a list of windows given an image, bounding box, window size and overlap fraction
    :param img: Original image
    :param x_start_stop: Tuple containing the start and stop positions of the whole sliding window area in the X dimension
    :param y_start_stop: Tuple containing the start and stop positions of the whole sliding window area in the Y dimension
    :param xy_window: Tuple containing the size of each sliding window
    :param xy_overlap: Overlap of each sliding window
    :return: A list of sliding windows described by a pair of two points: the top left and bottom right corner of the
             sliding window
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img_width
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img_height
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


def draw_boxes(img, bboxes, thick=6):
    """
    Draw all sliding windows defined by a given SlidingWindowAreaDefinition instance on a given image
    :param img: The original image
    :param bboxes: A list of bounding boxes
    :param thick: The thickness of the bounding box lines
    :return: The original image + bounding boxes drawn over it
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Colours for bounding boxes
    colours = [
        (255,0,0), (0,127,127), (0,255,0), (127,0,127), (0,0,255), (127,127,0), (0,0,0), (127,127,127), (255,255,255)]
    colour_idx = 0
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, (bbox[0][0]+(thick-1)//2, bbox[0][1]+(thick-1)//2), (bbox[1][0]-(thick-1)//2, bbox[1][1]-(thick-1)//2),
                      color=colours[colour_idx], thickness=thick)
        colour_idx += 1
        if colour_idx == len(colours):
            colour_idx = 0
    # Return the image copy with boxes drawn
    return imcopy


def draw_sliding_windows_on_test_images(windows):
    """
    Draw the sliding windows on a couple of test images
    :param windows: A list containing window definitions
    :return: Nothing
    """
    image1 = imread('test_images/test3.jpg')
    image2 = imread('test_images/test1.jpg')
    image3 = imread('test_images/test5.jpg')
    image4 = imread('test_images/test7.jpg')
    image5 = imread('test_images/test8.jpg')
    window_img1 = draw_boxes(image1, windows, color=(0, 0, 255), thick=6)
    window_img2 = draw_boxes(image2, windows, color=(0, 0, 255), thick=6)
    window_img3 = draw_boxes(image3, windows, color=(0, 0, 255), thick=6)
    window_img4 = draw_boxes(image4, windows, color=(0, 0, 255), thick=6)
    window_img5 = draw_boxes(image5, windows, color=(0, 0, 255), thick=6)

    plt.figure()
    plt.imshow(window_img1)
    plt.figure()
    plt.imshow(window_img2)
    plt.figure()
    plt.imshow(window_img3)
    plt.figure()
    plt.imshow(window_img4)
    plt.figure()
    plt.imshow(window_img5)
    plt.show(block=True)


if __name__ == "__main__":
    windows1 = slide_window(img_width=1280, img_height=720,
                            x_start_stop=(550, 1024),
                            y_start_stop=(370, 498),
                            xy_window=(64, 64),
                            xy_overlap=(0.5, 0.5))

    draw_sliding_windows_on_test_images(windows=windows1)


    windows2 = slide_window(img_width=1280, img_height=720,
                            x_start_stop=(530, 1144),
                            y_start_stop=(390, 534),
                            xy_window=(96, 96),
                            xy_overlap=(0.5, 0.5))

    draw_sliding_windows_on_test_images(windows=windows2)


    windows3 = slide_window(img_width=1280, img_height=720,
                            x_start_stop=(480, 1280),
                            y_start_stop=(400, 592),
                            xy_window=(128, 128),
                            xy_overlap=(0.5, 0.5))

    draw_sliding_windows_on_test_images(windows=windows3)


    windows4 = slide_window(img_width=1280, img_height=720,
                            x_start_stop=(944, 1280),
                            y_start_stop=(380, 620),
                            xy_window=(192, 160),
                            xy_overlap=(0.75, 0.75))

    draw_sliding_windows_on_test_images(windows=windows4)

    windows5 = slide_window(img_width=1280, img_height=720,
                            x_start_stop=(896, 1280),
                            y_start_stop=(396, 636),
                            xy_window=(256, 192),
                            xy_overlap=(0.75, 0.75))

    draw_sliding_windows_on_test_images(windows=windows5)