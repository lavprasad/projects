from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.metrics import Accuracy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing

# Training Image preprocessing
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = training_datagen.flow_from_directory(
    'C:/Users/Asus/Downloads/Object detection/Fruits-36dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Validation Image Preprocessing
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_set = validation_datagen.flow_from_directory(
    'C:/Users/Asus/Downloads/Object detection/Fruits-36dataset/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Building Model
cnn = Sequential()

# Building Convolution Layer
cnn.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(Dropout(0.5))  # To avoid overfitting
cnn.add(Flatten())
cnn.add(Dense(units=256, activation='relu'))
# Output Layer
cnn.add(Dense(units=36, activation='softmax'))

# Compiling and Training Phase
cnn.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[Accuracy()])
# cnn.summary()

training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=32)

# Evaluating Model

# Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)

# Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)

# Saving Model
cnn.save('C:/Users/Asus/Downloads/Object detection/trained_model.h5')
