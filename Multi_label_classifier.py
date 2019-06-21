import tensorflow
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
import os
import matplotlib.pyplot as plt
from keras import optimizers

history = None


#############image data generator###########
class Multi_label_image_CNN(object):
    img_rows, img_cols, img_ch = 256, 256, 3
    batch_size = 40
    nepochs = 10

    def __init__(self):
        print("Multi label image classification by CNN ")

    def read_data(self):
        dir_tr = 'C:\\Users\\Sajjad\\OneDrive\\Deep\\HWs\\HW3\\data1\\train\\'
        dir_val = 'C:\\Users\\Sajjad\\OneDrive\\Deep\\HWs\\HW3\\data1\\validation\\'

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
            class_mode='categorical')

        self.validation_generator = test_datagen.flow_from_directory(
            dir_val,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical')
        print(self.validation_generator.samples)
        #######images and Labels###############
        self.images, self.labels = next(self.train_generator)

    def getclasslabel(val):
        if val == 2:
            return "Person"
        else:
            return "No_Person"

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
        img_width, img_height = 256, 256
        cnn4 = Sequential()
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        cnn4.add(Conv2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
        cnn4.add(MaxPooling2D(2, 2))
        # The second convolution
        cnn4.add(Conv2D(32, (3, 3), activation='relu'))
        cnn4.add(MaxPooling2D(2, 2))
        # The third convolution
        cnn4.add(Conv2D(64, (3, 3), activation='relu'))
        cnn4.add(MaxPooling2D(2, 2))
        # The fourth convolution
        cnn4.add(Conv2D(64, (3, 3), activation='relu'))
        cnn4.add(MaxPooling2D(2, 2))
        # The fifth convolution
        cnn4.add(Conv2D(64, (3, 3), activation='relu'))
        cnn4.add(MaxPooling2D(2, 2))
        # Flatten the results to feed into a DNN
        cnn4.add(Flatten())
        # 512 neuron hidden layer
        cnn4.add(Dense(512, activation='relu'))

        cnn4.add(Dense(256, activation='relu'))

        cnn4.add(Dense(3, activation='softmax'))

        cnn4.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
        cnn4.summary()
        return cnn4

    def fit_model(self, classifier):
        # print(self.validation_generator.samples// self.batch_size)

        checkpointer = MyEarly_Stopper()

        history = classifier.fit_generator(
            self.train_generator,
            steps_per_epoch=((self.train_generator.samples)) // self.batch_size,
            epochs=self.nepochs, verbose=1,
            validation_data=self.validation_generator,
            validation_steps=((self.validation_generator.samples)) // self.batch_size, callbacks=[checkpointer])
        return history

    def plot_accuracy(self, history, miny=None):
        acc = history.history['acc']

        test_acc = history.history['val_acc']

        nepochs = range(len(acc))

        plt.plot(nepochs, acc, label='Acc')
        plt.plot(nepochs, test_acc, label='Test_Acc')
        if miny:
            plt.ylim(miny, 1.0)
        plt.title('accuracy')
        plt.legend(loc='upper left')
        plt.ylim(0.3, 1.2)
        plt.show()

    def test(self,classifier):


        # predicting images
            path = path = 'C:\\Users\\sajjad\\OneDrive\\Deep\\HWs\\HW3\\data1\\imagescc.jpg'
            img = image.load_img(path, target_size=(self.img_rows, self.img_cols))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = classifier.predict(images, batch_size=16)
            print(classes[0])
            if classes[0][2] > 0.5:
                print("It is a human image")
            else:
                print("It is a fruit image")
accuracy_threshold=0.75

class MyEarly_Stopper(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > accuracy_threshold):
            print("\nMaximum accuracy has reached, so stop training!!" + str((accuracy_threshold * 100)))
            self.model.stop_training = True

def main():

    cnn = Multi_label_image_CNN()
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
        accuracy_threshold = float(input("Please enter accuracy threshold for earlystoping Training( A value in the range of [0.0, 1.0] )"))
    except:# by default callback is inactive, so initialize accuracy_threshold with 1.1
        accuracy_threshold = 1.10
    # Question 5. compile and fit model
    history = cnn.fit_model(classifier)
    cnn.plot_accuracy(history, miny=0.95)

    # Question 6. Validate the model using an optional image
    cnn.test(classifier)

if __name__ == '__main__':
    main()

