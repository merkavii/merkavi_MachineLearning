import os
import shutil
import random

source_dir = 'PetImages'
base_dir = 'data'

# reset data folder
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

for split in ['train', 'test']:
    for cls in ['Cat', 'Dog']:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

def split_data(class_name, split_ratio=0.8):
    files = os.listdir(os.path.join(source_dir, class_name))
    files = [f for f in files if f.lower().endswith('.jpg')]
    random.shuffle(files)

    split_point = int(len(files) * split_ratio)

    for f in files[:split_point]:
        shutil.copy(
            os.path.join(source_dir, class_name, f),
            os.path.join(train_dir, class_name, f)
        )

    for f in files[split_point:]:
        shutil.copy(
            os.path.join(source_dir, class_name, f),
            os.path.join(test_dir, class_name, f)
        )

split_data('Cat')
split_data('Dog')

print("Dataset split done correctly ✅")