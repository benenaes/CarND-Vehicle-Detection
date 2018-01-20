import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread


def find_minimum_error_patch(new_img, patch):
    """
    Find the location inside a given image where there is a minimum matching error with a given patch
    :param new_img: The image to look in for a match
    :param patch: The patch to match
    :return: The minimum matching error and the location of the top left corner of the match
    """
    templ_sqdiff = cv2.matchTemplate(new_img, patch, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(templ_sqdiff)
    return min_val, min_loc


def find_unique_cars(imgs, max_diff):
    """
    Find a series of unique cars inside a list of images
    :param imgs: The list of images
    :param max_diff: The max matching error to be regarded as a non unique car image
    :return: A list of "unique" car images
    """
    first_img = imread(imgs[0])
    current_patch = first_img[16:48,16:48]
    unique_cars = [imgs[0]]
    for img in imgs:
        new_img = imread(img)
        diff1, max_loc1 = find_minimum_error_patch(new_img=new_img, patch=current_patch)
        if diff1 > max_diff:
            current_patch = new_img[16:48,16:48]
            unique_cars.append(img)
    return unique_cars


def find_unique_gti():
    """
    Find a list of images inside the GTI car database that have (semi) unique cars
    :return: A list of unique GTI car images
    """
    unique_cars = []
    gti_far = glob.glob("vehicles/GTI_Far/*.png")
    unique_cars.extend(find_unique_cars(gti_far, max_diff=0.05))
    gti_left = glob.glob("vehicles/GTI_Left/*.png")
    unique_cars.extend(find_unique_cars(gti_left, max_diff=0.10))
    gti_middle = glob.glob("vehicles/GTI_MiddleClose/*.png")
    unique_cars.extend(find_unique_cars(gti_middle, max_diff=0.05))
    gti_right = glob.glob("vehicles/GTI_Right/*.png")
    unique_cars.extend(find_unique_cars(gti_right, max_diff=0.10))
    return unique_cars


if __name__ == "__main__":
    orig_car = imread("vehicles/GTI_MiddleClose/image0000.png")
    orig_car = orig_car[16:48,16:48]
    same_car = imread("vehicles/GTI_MiddleClose/image0001.png")
    diff1, max_loc1 = find_minimum_error_patch(new_img=same_car, patch=orig_car)
    cv2.rectangle(same_car, max_loc1, (max_loc1[0]+32, max_loc1[1]+32), color=(0, 0, 255), thickness=1)
    same_car2 = imread("vehicles/GTI_MiddleClose/image0002.png")
    diff2, max_loc2 = find_minimum_error_patch(new_img=same_car2, patch=orig_car)
    cv2.rectangle(same_car2, max_loc2, (max_loc2[0] + 32, max_loc2[1] + 32), color=(0, 0, 255), thickness=1)
    other_car = imread("vehicles/GTI_MiddleClose/image0038.png")
    diff3, max_loc3 = find_minimum_error_patch(new_img=other_car, patch=orig_car)
    cv2.rectangle(other_car, max_loc3, (max_loc3[0] + 32, max_loc3[1] + 32), color=(0, 0, 255), thickness=1)
    other_car2 = imread("vehicles/GTI_MiddleClose/image0078.png")
    diff4, max_loc4 = find_minimum_error_patch(new_img=other_car2, patch=orig_car)
    cv2.rectangle(other_car2, max_loc4, (max_loc4[0] + 32, max_loc4[1] + 32), color=(0, 0, 255), thickness=1)
    fig, axes = plt.subplots(1,5)
    axes[0].imshow(orig_car)
    axes[0].set_title("Original car")
    axes[1].imshow(same_car)
    axes[1].set_title("Same car, diff: {:4f}".format(diff1))
    axes[2].imshow(same_car2)
    axes[2].set_title("Same car, diff: {:4f}".format(diff2))
    axes[3].imshow(other_car)
    axes[3].set_title("Other car, diff: {:4f}".format(diff3))
    axes[4].imshow(other_car2)
    axes[4].set_title("Other car, diff: {:4f}".format(diff4))
    plt.show(block=True)