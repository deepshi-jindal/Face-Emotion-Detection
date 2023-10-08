
# import required packages
#CV2 imports the OpenCV library, which is commonly used for image and video processing tasks.
import cv2 
# It is a class that allows you to build neural networks layer by layer in a sequential manner.
from keras.models import Sequential
#a class for creating 2D convolutional layers in a neural network.
#a class for creating 2D max pooling layers, which downsample the input along the spatial dimensions.
#a class for creating fully connected layers in a neural network.
# a class for implementing dropout regularization, which randomly drops out a fraction of input units during training to prevent overfitting.
# a class for flattening the multi-dimensional output from previous layers into a 1D vector.
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
# an optimizer that implements the Adam optimization algorithm, which is widely used for training neural networks.
from keras.optimizers import Adam
#a class that generates batches of augmented/processed image data for training and validation
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
#flow_from_directory :-generates batches of preprocessed image data and their corresponding labels from a directory structure.
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
# Sequential model is a linear stack of layers, where you can add layers one by one in sequence.
emotion_model = Sequential()

#"activation" refers to the activation function applied to the output of a neuron or a layer.
#relu=rectified linear unit.It simply returns the input if it is positive, and if it is negative, it returns zero.
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

#The ReLU activation introduces non-linearity to the network, enabling it to learn complex patterns and features.
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
#'softmax' is used to convert the raw output values into a probability distribution, indicating the likelihood of each emotion category.
emotion_model.add(Dense(7, activation='softmax'))

#disables the use of OpenCL (Open Computing Language) in the OpenCV library
#OpenCL is a framework for parallel computing that allows programs to execute on heterogeneous platforms, including CPUs and GPUs.
cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=300,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')

