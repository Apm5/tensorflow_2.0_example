import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def load_data(PATH):
    train_images = np.load(PATH + '/x_train.npy')
    train_labels = np.load(PATH + '/y_train.npy')
    test_images = np.load(PATH + '/x_test.npy')
    test_labels = np.load(PATH + '/y_test.npy')

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize

    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    tf.config.gpu.set_per_process_memory_growth(enabled=True)  # gpu memory set

    (train_images, train_labels, test_images, test_labels) = load_data('MNIST')

    with tf.device('/gpu:1'):  # If no GPU, comment on this line
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        # show
        model.summary()

        # train
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, batch_size=32, epochs=5)

        # test
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(test_acc)
