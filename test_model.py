import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model


def compute_score(test_code, test_label):
    distances = []
    for code in learned_codes:
        np.linalg.norm(code - test_code)
        distances.append(test_code)
    distances = np.array(distances)
    distance_with_labels = np.stack(distances, y_train, axis=-1)
    sorted_distance_with_labels = np.sort(distance_with_labels)
    i = 0
    e = [i]
    while e[1] == test_label:
        i += 1
        e = sorted_distance_with_labels[i]
    return (i+1)/1000


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    autoencoder = load_model('mnist.h5')
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

    learned_codes = encoder.predict(x_train)
    test_codes = encoder.predict(x_test)
    test_code = test_codes[0]
    test_label = y_test[0]
    score = compute_score(test_code, test_label)
    print(score)
