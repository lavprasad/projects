from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

train_path = "C:/Users/Asus/Downloads/Object detection/fruits-4_dataset/fruits-4/Training/"
test_path = "C:/Users/Asus/Downloads/Object detection/fruits-4_dataset/fruits-4/Test/"

BatchSize = 32

# Build the model (adjust architecture as needed)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.4))  # Adjusted dropout rate
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))  # Adjusted dropout rate
model.add(Dense(3, activation="softmax"))  # Adjusted for 3 classes

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Load the data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into training and validation sets
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=BatchSize,
    class_mode="categorical",
    subset="training"  # Use training subset
)

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=BatchSize,
    class_mode="categorical",
    subset="validation"  # Use validation subset
)

# Early Stopping
stop_early = EarlyStopping(monitor="val_accuracy", patience=5)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[stop_early]
)

# Save the model
model.save("C:/Users/Asus/Downloads/Object detection/Fruits4_updated_WG_2.h5")
