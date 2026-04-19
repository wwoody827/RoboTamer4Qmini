import os
from os.path import exists
import shutil


def clear_dir(dir):
    if exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)


def safe_cv2_crop(img, crop_size=600):
    try:
        h, w = img.shape[:2]
        target_size = min(crop_size, w, h)

        x = (w - target_size) // 2 + 30
        y = (h - target_size) // 2 + min(crop_size, 100)

        if x < 0 or y < 0:
            print(f"Warning: image {w}x{h} smaller than crop {crop_size}")
            return None

        cropped = img[y:y + target_size, x:x + target_size - 100]
        return cropped

    except Exception as e:
        print(f"Error in safe_cv2_crop: {str(e)}")
        return None
