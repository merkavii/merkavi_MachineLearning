import tensorflow_datasets as tfds
from PIL import Image
import os

output_dir = 'cats_dogs_extracted'
os.makedirs(output_dir, exist_ok=True)
cat_dir = os.path.join(output_dir, 'cats')
dog_dir = os.path.join(output_dir, 'dogs')
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

dataset = tfds.load('cats_vs_dogs', split='train', as_supervised=True)

for i, (image, label) in enumerate(tfds.as_numpy(dataset)):
    img = Image.fromarray(image)
    if label == 0:
        img.save(os.path.join(cat_dir, f'cat_{i}.jpg'))
    else:
        img.save(os.path.join(dog_dir, f'dog_{i}.jpg'))

print("Extraction done!")