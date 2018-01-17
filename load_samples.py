import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from random import shuffle

from find_unique_gti import find_unique_gti


def get_kitti_car_images():
    kitti = glob.glob("vehicles/KITTI_extracted/*.png")
    return kitti


def get_non_vehicle_data():
    gti = glob.glob("non-vehicles/GTI/*.png")
    extras = glob.glob("non-vehicles/Extras/*.png")
    total = np.concatenate((gti, extras))
    return total


if __name__ == "__main__":
    kitti = get_kitti_car_images()
    gti = find_unique_gti()
    vehicles = np.concatenate(kitti, gti)
    shuffle(vehicles)
    non_vehicles = get_non_vehicle_data()
    shuffle(non_vehicles)
    print("Number of vehicles samples: %d" % len(vehicles))
    print("Number of non-vehicle samples: %d" % len(non_vehicles))

    fig, axes = plt.subplots(2, 4)
    for idx in range(4):
        sub_plot = axes[0][idx]
        sub_plot.axis('off')
        sub_plot.set_title('Vehicle sample')
        img = imread(vehicles[idx])
        sub_plot.imshow(img)
        sub_plot = axes[1][idx]
        sub_plot.axis('off')
        sub_plot.set_title('Non-vehicle sample')
        img = imread(non_vehicles[idx])
        sub_plot.imshow(img)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show(block=True)

