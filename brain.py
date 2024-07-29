import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

image_dir = r'C:\Users\kkrav\OneDrive\Documents\data_science projects\brain tumour\brain_tumor_dataset'
no_tumor_images_dir = os.path.join(image_dir, 'no')
yes_tumor_images_dir = os.path.join(image_dir, 'yes')

dataset = []
label = []
INPUT_SIZE = 64

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image at {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    return np.array(image)

for image_name in os.listdir(no_tumor_images_dir):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(no_tumor_images_dir, image_name)
        image_array = process_image(image_path)
        if image_array is not None:
            dataset.append(image_array)
            label.append(0)

for image_name in os.listdir(yes_tumor_images_dir):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(yes_tumor_images_dir, image_name)
        image_array = process_image(image_path)
        if image_array is not None:
            dataset.append(image_array)
            label.append(1)

dataset = np.array(dataset)
label = np.array(label)

# Print data balance
print(f"Number of 'No Tumor' images: {np.sum(label == 0)}")
print(f"Number of 'Yes Tumor' images: {np.sum(label == 1)}")

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=True)

model.save('brain_tumor.h5')

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')
