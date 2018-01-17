import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread


def find_minimum_error_patch(new_img, original_img):
    templ_sqdiff = cv2.matchTemplate(new_img, original_img, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(templ_sqdiff)
    return min_val, min_loc


def find_unique_cars(imgs, max_diff):
    first_img = imread(imgs[0])
    current_patch = first_img[16:48,16:48]
    unique_cars = [imgs[0]]
    for img in imgs:
        new_img = imread(img)
        diff1, max_loc1 = find_minimum_error_patch(new_img=new_img, original_img=current_patch)
        if diff1 > max_diff:
            current_patch = new_img[16:48,16:48]
            unique_cars.append(img)
    return unique_cars


def find_unique_gti():
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
    unique_cars = find_unique_gti()
    print(len(unique_cars))
    print('\n'.join(unique_cars))
    # orig_car = imread("vehicles/GTI_Left/image0009.png")
    # orig_car = orig_car[16:48,16:48]
    # same_car = imread("vehicles/GTI_Left/image0012.png")
    # diff1, max_loc1 = find_minimum_error_patch(new_img=same_car, original_img=orig_car)
    # cv2.rectangle(same_car, max_loc1, (max_loc1[0]+32, max_loc1[1]+32), color=(0, 0, 255), thickness=1)
    # same_car2 = imread("vehicles/GTI_Left/image0014.png")
    # diff2, max_loc2 = find_minimum_error_patch(new_img=same_car2, original_img=orig_car)
    # cv2.rectangle(same_car2, max_loc2, (max_loc2[0] + 32, max_loc2[1] + 32), color=(0, 0, 255), thickness=1)
    # other_car = imread("vehicles/GTI_Left/image0019.png")
    # diff3, max_loc3 = find_minimum_error_patch(new_img=other_car, original_img=orig_car)
    # cv2.rectangle(other_car, max_loc3, (max_loc3[0] + 32, max_loc3[1] + 32), color=(0, 0, 255), thickness=1)
    # other_car2 = imread("vehicles/GTI_Left/image0022.png")
    # diff4, max_loc4 = find_minimum_error_patch(new_img=other_car2, original_img=orig_car)
    # cv2.rectangle(other_car2, max_loc4, (max_loc4[0] + 32, max_loc4[1] + 32), color=(0, 0, 255), thickness=1)
    # fig, axes = plt.subplots(1,5)
    # axes[0].imshow(orig_car)
    # axes[0].set_title("Original car")
    # axes[1].imshow(same_car)
    # axes[1].set_title("Same car, diff: {:4f}".format(diff1))
    # axes[2].imshow(same_car2)
    # axes[2].set_title("Same car, diff: {:4f}".format(diff2))
    # axes[3].imshow(other_car)
    # axes[3].set_title("Other car, diff: {:4f}".format(diff3))
    # axes[4].imshow(other_car2)
    # axes[4].set_title("Other car, diff: {:4f}".format(diff4))
    # plt.show(block=True)