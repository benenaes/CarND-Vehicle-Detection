from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from random import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pickle

from calculate_features import extract_features, HogParameters
from load_samples import get_non_vehicle_data, get_vehicle_data


def train_svm(train_features, train_labels):
    clf = SVC(C=10.0, kernel="rbf", gamma=0.0001, probability=True)
    clf.fit(train_features, train_labels)
    return clf


def search_svm_parameters(train_features, train_labels, test_features, test_labels):
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


def prepare_training_data(
        colour_space='HSV',
        orientations=18,
        pix_per_cell=8,
        cell_per_block=2,
        hog_channel='ALL',
        spatial_size=(16,16),
        hist_bins=32):
    """
    Prepare the HOG data for training/testing the classifier
    :param colour_space: Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    :param orientations:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel: Can be 0, 1, 2, or "ALL"
    :param spatial_size:
    :param hist_bins:
    :return:
    """
    vehicles = get_vehicle_data()
    shuffle(vehicles)
    non_vehicles = get_non_vehicle_data()
    shuffle(non_vehicles)
    car_features = extract_features(
        imgs=vehicles,
        colour_space=colour_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        hog_parameters=HogParameters(orientations=orientations, pixels_per_cell=pix_per_cell, cells_per_block=cell_per_block),
        hog_channel=hog_channel)
    not_car_features = extract_features(
        imgs=non_vehicles,
        colour_space=colour_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        hog_parameters=HogParameters(orientations=orientations, pixels_per_cell=pix_per_cell, cells_per_block=cell_per_block),
        hog_channel=hog_channel)

    # Create an array stack of feature vectors
    all_features = np.vstack((car_features, not_car_features)).astype(np.float64)
    # Fit a per-column scaler
    feature_scaler = StandardScaler().fit(all_features)
    # Apply the scaler to X
    scaled_features = feature_scaler.transform(all_features)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_features, y, test_size=0.2, random_state=rand_state)

    return x_train, x_test, y_train, y_test, feature_scaler


def grid_search_svc_parameters_with_training_data():
    """
    Grid search of SVM classifier parameters to find a good parameter for the kernel, gamma and C parameters.
    First, an optimization for precision is started and stored in the file "precision-svm.p"
    Second, an optimization for recall is started and stored in the file "recall-svm.p"
    :return: Nothing. The
    """
    x_train, x_test, y_train, y_test, feature_scaler = prepare_training_data(colour_space='HSV')
    clfs = search_svm_parameters(x_train, y_train, x_test, y_test)
    print('Test accuracy of precision optimized SVM = ', round(clfs[0].score(x_test, y_test), 4))
    print('Test accuracy of recall optimized SVM = ', round(clfs[1].score(x_test, y_test), 4))
    with open('precision-svm.p', 'wb') as pickle_file:
        pickle.dump(clfs[0], pickle_file)
    with open('recall-svm.p', 'wb') as pickle_file:
        pickle.dump(clfs[1], pickle_file)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, feature_scaler = prepare_training_data(
        colour_space='HSV', orientations=18, pix_per_cell=8, cell_per_block=2, hog_channel='ALL')
    clf = train_svm(x_train, y_train)
    print('Test Accuracy of SVM = ', round(clf.score(x_test, y_test), 4))
    with open('all-features-rbf-svm.p', 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)
    with open('all-features-scaler.p', 'wb') as pickle_file:
        pickle.dump(feature_scaler, pickle_file)