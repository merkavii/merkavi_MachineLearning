from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# لود مدل ذخیره‌شده
model = load_model(r"D:/machine_learning/deep_learning/cats_dogs_model.keras")

while True:
    img_path = input("Enter image path (or type exit): ")

    if img_path.lower() == "exit":
        break

    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    result = model.predict(img)
    
    if result[0][0] > 0.5:
        print("🐶 Dog")
    else:
        print("🐱 Cat")