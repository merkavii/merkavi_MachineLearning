import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# Part1 - data Preprocessing
#1_Preprocessing the training set
#  یسری تغیرات باید روی عکس ها اعمال شه روی پیکسلاش و ... برای جلوگیری از اور فیتینگ و ایناهم هست
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

training_set = train_datagen.flow_from_directory(
    'data/train',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary'
)

#2_Preprocessing the test set

test_datagen = ImageDataGenerator(
    rescale = 1./255,
)

test_set = test_datagen.flow_from_directory(
    'data/test',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary'
)

#Part2 - Building the CNN
# 1_initialising the CNN
cnn = tf.keras.models.Sequential()

# 2_convolution
cnn.add(tf.keras.layers.Conv2D(filters=32 , kernel_size= 3, activation= 'relu', input_shape=[64,64,3]))
# add برای اضافه کردن لایه
# filter = feature Detector . همونی که با عکس ورودی ضرب میشد لایه کانولوشن میساخت
# kernel_size : اندازه فیلتر ما(اینجا ما میخوایم 3 در 3 باشه)
# activation : تابع فعال سازی که باعث میشد داده ما از حالت خطی در بیاد
# input_shape : شکل ورودیمون چیه.از اونجایی که ورودی ما رنگیه همون ار جی بی اینو وارد میکنیم و اندازه تصویرمونو چند خط بالا به 64 به 64 تغییر دادیم اینو میزنیم

# 3_pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size =2 ,strides = 2))
# ba ravesh Maxpooling va andaze pool,pooling mikonim
# strides : دو پیکسل دو پیکسل جلو بره و مکس پولینگ روش اعمال شه

#4_Adding a second convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32 , kernel_size= 3, activation= 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size =2 ,strides = 2))

# 5_Flattening
cnn.add(tf.keras.layers.Flatten())

# 6_full connection
cnn.add(tf.keras.layers.Dense(units = 128 ,activation = 'relu'))

# 7_output layer
cnn.add(tf.keras.layers.Dense(units = 1 ,activation = 'sigmoid'))

# Part3 - training the CNN
# 1_compiling the CNN
cnn.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

# 2_training the CNN on the training set and evaluating it on the test set
# evaluating : ارزیابی
cnn.fit(x = training_set,validation_data = test_set, epochs = 25)

#Part4 - Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('D:\machine_learning\deep_learning\data\predict\dog1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
# دلیل سه بعدی بودن اینه که مدل سی ان ان ما با دسته های 32 تایی یاد گرفته و دسته ها توی بعد سوم بودن پس ما باید به عکسمون یه بعد اضافه کنیم
result = cnn.predict(test_image)
print(training_set.class_indices) # برای اینکه بفهمیم منظور از 0 و 1 کدوم سگ و کدوم گربس

if result[0][0] == 1:
    prediction = 'Dog'
else :
    prediction = 'Cat'
print(prediction)
cnn.save('cats_dogs_model.keras')