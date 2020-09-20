import itertools
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_directory = os.path.join(os.path.dirname(__file__), "dataset")
model_save_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
EPOCH_VAL = 10


def get_image_data(dataset_directory):
    image_data = []
    for dirname, _, filenames in os.walk(dataset_directory):
        type_of_image = (str(dirname).split('/')[-1])
    for filename in filenames:
        # print(os.path.join(dirname,filename))
        img = image.load_img(os.path.join(dirname, filename), target_size=(256, 256))
        image_array = image.img_to_array(img) / 255
        image_data.append((image_array, type_of_image))
        # directory of loaded image
        print("processing image: " + str(dirname) + " / " + str(filename))

    print("data is loaded")
    print("number of trainning images =", end=" ")
    print(len(image_data))
    return image_data


def generate_training_test_data(image_data):
    labels = {"no": 0, "yes": 1}
    random.seed(1)
    # random shuffling of image data
    random.shuffle(image_data)
    train = image_data[:int(0.90 * len(image_data))]
    validation = image_data[int(0.90 * len(image_data)):]

    # In[9]:

    # Spliting of x_train and y_train
    x_train = np.asarray([data[0] for data in train])
    y_train_decoded = np.asarray([labels[data[1]] for data in train])
    y_train = tf.keras.utils.to_categorical(y_train_decoded)  # encoding to one hot vector
    # Spliting of x_val and y_val
    x_val = np.asarray([data[0] for data in validation])
    y_val_decoded = np.asarray([labels[data[1]] for data in validation])
    y_val = tf.keras.utils.to_categorical(y_val_decoded)  # encoding to one hot vector
    print(len(x_train))
    print(len(y_train))
    print(len(x_val))
    print(len(y_val))
    return (x_train, y_train_decoded, y_train, x_val, y_val_decoded, y_val)


def print_plot_data(x_train, y_train_decoded, y_train, x_val, y_val_decoded, y_val, image_data):
    plt.hist(y_train_decoded)
    plt.plot()
    print(Counter(y_train_decoded))
    plt.hist(y_val_decoded)
    plt.plot()
    print(Counter(y_val_decoded))

    # printing a trainning image
    plt.imshow(image_data[8][0])
    print(image_data[8][1])


def generate_heruistics():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    return model, datagen


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def model_fit(model, datagen, x_train, y_train_decoded, y_train, x_val, y_val_decoded, y_val, image_data):
    datagen.fit(x_train)

    # In[14]:

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    # In[ ]:

    history = model.fit(datagen.flow(x_train, y_train, batch_size=86),
                        epochs=EPOCH_VAL, validation_data=(x_val, y_val))

    model.save(model_save_directory)
    print("model is saved")


def test_prediction(model, x_train, y_train_decoded, y_train, x_val, y_val_decoded, y_val, image_data):
    # Predict the values from the validation dataset
    Y_pred = model.predict(x_val)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(y_val, axis=1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes=range(2))

    # In[ ]:

    test_loss, test_acc = model.evaluate(x_val, y_val, verbose=1)


if __name__ == '__main__':
    image_data = get_image_data(dataset_directory)
    x_train, y_train_decoded, y_train, x_val, y_val_decoded, y_val = generate_training_test_data(image_data)
    model, datagen = generate_heruistics()
    model_fit(model, datagen, x_train, y_train_decoded, y_train, x_val, y_val_decoded, y_val, image_data)
