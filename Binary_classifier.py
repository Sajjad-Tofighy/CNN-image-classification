import keras

from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
import os

import numpy as np
import random

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pylab as plt

history = None


#############image data generator###########
class Binary_image_CNN(object):

    def __init__(self):
        self.history = None
        self.nepochs = 5
        self.batch_size = 40
        self.img_rows, self.img_cols, self.img_ch = 256, 256, 3
        print("Binary image classification by CNN ")

    def read_data(self):
        dir_tr = 'C:\\Users\\sajjad\\OneDrive\\Deep\\HWs\\HW3\\data\\train\\'
        dir_val = 'C:\\Users\\sajjad\\OneDrive\\Deep\\HWs\\HW3\data\\validation\\'

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            dir_tr,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='binary')

        self.validation_generator = test_datagen.flow_from_directory(
            dir_val,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='binary')
        print(self.validation_generator.samples)
        #######images and Labels###############
        self.images, self.labels = next(self.train_generator)

    def show_image(self):
        n = 8  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.images[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.title.set_text(self.labels[i])
        plt.show()

    def model_design(self):
        input_img = Input(shape=(self.img_rows, self.img_cols, self.img_ch))

        classifier = Sequential()
        classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                              input_shape=(self.img_rows, self.img_cols, self.img_ch)))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(MaxPooling2D(pool_size=(3, 3)))
        classifier.add(Conv2D(128, (3, 3), activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(MaxPooling2D(pool_size=(3, 3)))

        classifier.add(Flatten())
        classifier.add(Dense(64))
        classifier.add(Activation('relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(1))
        classifier.add(Activation('sigmoid'))

        classifier.summary()
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return classifier

    import tensorflow
    accuracy_threshold = 0.7

    def fit_model(self, classifier):
        print(self.validation_generator.samples // self.batch_size)

        checkpointer = MyEarly_Stopper()

        self.history = classifier.fit_generator(
            self.train_generator,
            steps_per_epoch=((self.train_generator.samples)) // self.batch_size,
            epochs=self.nepochs, verbose=1,
            validation_data=self.validation_generator,
            validation_steps=((self.validation_generator.samples)) // self.batch_size, callbacks=[checkpointer])
        return self.history

    def plot_accuracy(self):
        acc = self.history.history['acc']

        test_acc = self.history.history['val_acc']

        nepochs = range(len(acc))

        plt.plot(nepochs, acc, label='Acc')
        plt.plot(nepochs, test_acc, label='Test_Acc')

        plt.title('accuracy')
        plt.legend(loc='upper left')
        plt.ylim(0.3, 1.2)
        plt.show()

    def visualize_layer(self, layer_names, successive_feature_maps):
        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
            if len(feature_map.shape) == 4:
                # Just do this for the conv / maxpool layers, not the fully-connected layers
                n_features = feature_map.shape[-1]  # number of features in feature map
                # The feature map has shape (1, size, size, n_features)
                size = feature_map.shape[1]
                # We will tile our images in this matrix
                display_grid = np.zeros((size, size * n_features))
                for i in range(n_features):
                    # Postprocess the feature to make it visually palatable
                    x = feature_map[0, :, :, i]
                    x -= x.mean()
                    x /= x.std()
                    x *= 64
                    x += 128
                    x = np.clip(x, 0, 255).astype('uint8')
                    # We'll tile each filter into this big horizontal grid
                    display_grid[:, i * size: (i + 1) * size] = x
                # Display the grid
                scale = 20. / n_features
                plt.figure(figsize=(scale * n_features, scale))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')

    def test(self,classifier):


        # predicting images
            path = path = 'C:\\Users\\sajjad\\OneDrive\\Deep\\HWs\\HW3\\data1\\imagescc.jpg'
            img = image.load_img(path, target_size=(self.img_rows, self.img_cols))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = classifier.predict(images, batch_size=16)
            print(classes[0])
            if classes[0] > 0.5:
                print("It is a human image")
            else:
                print("It is a fruit image")


class MyEarly_Stopper(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > accuracy_threshold):
            print("\nMaximum accuracy has reached, so stop training!!" + str((accuracy_threshold * 100)))
            self.model.stop_training = True


def visualize(cnn, classifier):
    # train_horse_names = os.listdir(train_horse_dir)
    person_dirname = os.path.join('C:\\Users\\Sajjad\\OneDrive\\Deep\\HWs\\HW3\\data\\validation\\person\\')
    fruit_dirname = os.path.join('C:\\Users\\Sajjad\\OneDrive\\Deep\\HWs\\HW3\\data\\validation\\fruit\\')

    person_dir = os.listdir('C:\\Users\\Sajjad\\OneDrive\\Deep\\HWs\\HW3\\data\\validation\\person\\')
    fruit_dir = os.listdir('C:\\Users\\Sajjad\\OneDrive\\Deep\\HWs\\HW3\\data\\validation\\fruit\\')

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.

    successive_outputs = [layer.output for layer in classifier.layers[0:]]

    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = Model(inputs=classifier.input, outputs=successive_outputs)

    # Let's prepare a random input image from the training set.

    fruit_img_files = [os.path.join(fruit_dirname, f) for f in fruit_dir]
    human_img_files = [os.path.join(person_dirname, f) for f in person_dir]
    person_img_path = random.choice(human_img_files)
    fruit_img_path = random.choice(fruit_img_files)

    pimg = load_img(person_img_path, target_size=(256, 256))  # this is a PIL image
    px = img_to_array(pimg)  # Numpy array with shape (150, 150, 3)
    px = px.reshape((1,) + px.shape)  # Numpy array with shape (1, 150, 150, 3)

    fimg = load_img(fruit_img_path, target_size=(256, 256))  # this is a PIL image
    fx = img_to_array(fimg)  # Numpy array with shape (150, 150, 3)
    fx = fx.reshape((1,) + fx.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    px /= 255
    fx /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(px)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in classifier.layers]

    # Now let's display our representations

    cnn.visualize_layer(layer_names, successive_feature_maps)

    # print()

    # print("++++++++++++++++++ Visualize Cnovnet output of a Fruit image++++++++++++++++++++++++++++++++ ")
    successive_feature_maps = visualization_model.predict(fx)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in classifier.layers]

    cnn.visualize_layer(layer_names, successive_feature_maps)

def main():
    cnn = Binary_image_CNN()
    # Question 1. Read data using ImageGenerator and show
    # n=2 randomly choosed images
    cnn.read_data()
    cnn.show_image()
    # Question 2. Configure selected architucture for CNN network
    # and show summary.
    classifier = cnn.model_design()
    # Question 4. callback threshold setup
    global accuracy_threshold
    try:
        accuracy_threshold=float(input("Please enter accuracy threshold for earlystoping Training( A value in tha range of [0.0, 1.0] )"))
    except:# by default callback is inactive, so initialize accuracy_threshold with 1.1
        accuracy_threshold=1.10
    # Question 5. compile and fit model
    history = cnn.fit_model(classifier)
    cnn.plot_accuracy()
    # Question 6. Validate the model for an optional image
    cnn.test(classifier)
    # Question 7. show output of intermediate CNN layers
    if accuracy_threshold==1.0:
        visualize(cnn, classifier)
    #


if __name__ == '__main__':
    main()
