import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cv2 import imread,resize


# Define the input shape
input_shape = (60, 60, 3)

# Define the number of classes
num_classes = 3


train_img = []

for i in range(1,200):
    try:
            image_path = 'train/'+ str(i)+'.jpg' 
            #print(image_path)
            img = imread(image_path)
            img = img/255 #Normalizing pixels into 0-1
            img = resize(img, (60,60)) #This can be resized to any shape, choosing 224 randomly. Larger sizes can result in computation delay
            train_img.append(img)
    except:
            pass

train_x = np.array(train_img)

    # defining the labels
train_y1 = np.tile([1, 0, 0], (55, 1))
train_y2 = np.tile([0, 1, 0], (34, 1))
train_y3 = np.tile([0, 0, 1], (98, 1))
train_y = np.vstack((train_y1,train_y2,train_y3))
print(train_y.shape,train_x.shape)


train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=13, stratify=train_y) #80:20 Train/Test split

# Define the model architecture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, batch_size=5, epochs=5, validation_data=(test_x, test_y))

model.save('eye.h5')

'''
model = keras.models.load_model('eye.h5')

img = imread('train/65.jpg')
img = img/255 #Normalizing pixels into 0-1
img = resize(img, (60,60)) #This can be resized to any shape, choosing 224 randomly. Larger sizes can result in computation delayprint(model.predict(img))

test = []
test.append(img)
test = np.array(test)
out = np.argmax(model.predict(test))
if(out == 0):
        print('right')
elif(out == 0):
        print('left')
else:
        print('center')


'''