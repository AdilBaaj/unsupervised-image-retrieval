import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score
import time


def compute_score_1(test_code, test_label):
    distances = []
    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    distances = np.array(distances)
    distance_with_labels = np.stack((distances, y_train), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]
    # sorted_labels = sorted_distance_with_labels[:, 1]
    # k = np.where(sorted_labels == test_label)
    i = 0
    e = sorted_distance_with_labels[i][1]
    while e == test_label:
        i = i + 1
        e = sorted_distance_with_labels[i][1]
    return i+1


def compute_average_precision_score(test_codes, test_labels):
    out_labels = []
    out_distances = []
    for i in range(len(test_codes)):
        distances = []
        for code in learned_codes:
            distance = np.linalg.norm(code - test_codes[i])
            distances.append(distance)
        distances = np.array(distances)
        out_distances.append(distances)
        labels = np.copy(y_train).astype('float32')
        labels[labels != test_labels[i]] = -1
        labels[labels == test_labels[i]] = 1
        labels[labels == -1] = 0
        out_labels.append(labels)

    out_labels = np.array(out_labels)
    out_distances = np.array(out_distances)
    score = label_ranking_average_precision_score(out_labels, out_distances)
    return score


if __name__ == '__main__':
    print('Loading mnist dataset')
    t0 = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    t1 = time.time()
    print('mnist dataset loaded in: ', t1-t0)

    print('Loading model :')
    t0 = time.time()
    autoencoder = load_model('mnist.h5')
    t1 = time.time()
    print('Model loaded in: ', t1-t0)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

    learned_codes = encoder.predict(x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    learned_codes = learned_codes.reshape(learned_codes.shape[0], learned_codes.shape[1])
    test_codes = encoder.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1])
    indexes = np.random.randint(test_codes.shape[0], size=100)

    print('Start computing score')
    t1 = time.time()
    score = compute_average_precision_score(test_codes[indexes], y_test[indexes])
    t2 = time.time()
    print('Score computed in: ', t2-t1)
    print('Model score:', score)
