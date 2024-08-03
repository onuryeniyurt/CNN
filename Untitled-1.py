import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


train = ImageDataGenerator(rescale = 1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set = train.flow_from_directory('C:\\Users\\OnurY\\Desktop\\CNN\\Train', target_size=(64,64),batch_size=32,class_mode='binary')

test=ImageDataGenerator(rescale=1./255)
testing_set= test.flow_from_directory('C:\\Users\\OnurY\\Desktop\\CNN\\Test',target_size=(64,64),batch_size=32,class_mode='binary')


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu', input_shape=[64,64,3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=128,activation='relu'))

model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


model.fit(x=training_set,validation_data=testing_set,epochs=32)

test_image = image.load_img('C:\\Users\\OnurY\\Desktop\\CNN\\23.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'


print(prediction)