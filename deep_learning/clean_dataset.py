import os
from PIL import Image

def remove_corrupt_images(folder_path):
    removed_count = 0
    checked_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                checked_count += 1
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # بررسی سالم بودن فایل
                except Exception:
                    print(f"Removed corrupt file: {file_path}")
                    os.remove(file_path)
                    removed_count += 1

    print("\n========== REPORT ==========")
    print(f"Checked images: {checked_count}")
    print(f"Removed corrupt images: {removed_count}")
    print("Cleaning complete ✅")


if __name__ == "__main__":
    dataset_path = "data"   # یا "PetImages" اگر قبل از split میخوای تمیز کنی
    remove_corrupt_images(dataset_path)