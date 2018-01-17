from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from random import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pickle

from calculate_features import extract_hog_features, HogParameters
from load_samples import get_non_vehicle_data, get_kitti_car_images


def train_svm(train_features, train_labels, test_features, test_labels):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    clfs = []
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=4,
                           scoring='%s_macro' % score)
        clf.fit(train_features, train_labels)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_labels, clf.predict(test_features)
        print(classification_report(y_true, y_pred))
        print()
        clfs.append(clf)

    return clfs


def prepare_hog_data(colour_space='HSV', orientations=18, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    """
    Prepare the HOG data for training/testing the classifier
    :param colour_space: Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    :param orientations:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel: Can be 0, 1, 2, or "ALL"
    :return:
    """
    kitti = get_kitti_car_images()
    vehicles = kitti
    shuffle(vehicles)
    non_vehicles = get_non_vehicle_data()
    shuffle(non_vehicles)
    car_features = extract_hog_features(
        imgs=vehicles,
        colour_space=colour_space,
        hog_parameters=HogParameters(orientations=orientations, pixels_per_cell=pix_per_cell, cells_per_block=cell_per_block),
        hog_channel=hog_channel)
    not_car_features = extract_hog_features(
        imgs=non_vehicles,
        colour_space=colour_space,
        hog_parameters=HogParameters(orientations=orientations, pixels_per_cell=pix_per_cell, cells_per_block=cell_per_block),
        hog_channel=hog_channel)

    # Create an array stack of feature vectors
    hog_features = np.vstack((car_features, not_car_features)).astype(np.float64)
    # Fit a per-column scaler
    hog_scaler = StandardScaler().fit(hog_features)
    # Apply the scaler to X
    scaled_hogs = hog_scaler.transform(hog_features)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_hogs, y, test_size=0.2, random_state=rand_state)

    return x_train, x_test, y_train, y_test, hog_scaler


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, hog_scaler = prepare_hog_data(
        colour_space='HSV', orientations=18, pix_per_cell=8, cell_per_block=2, hog_channel='ALL')
    clfs = train_svm(x_train, y_train, x_test, y_test)
    print('Test Accuracy of precision optimized SVM = ', round(clfs[0].score(x_test, y_test), 4))
    print('Test Accuracy of recall optimized SVM = ', round(clfs[1].score(x_test, y_test), 4))
    pickle.dump(clfs[0], "precision-svm.p")
    pickle.dump(clfs[1], "recall-svm.p")
    pickle.dump(hog_scaler, "hog-scaler.p")