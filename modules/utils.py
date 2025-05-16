from PIL import Image
import numpy as np

def load_image(image_file, target_size=(512, 512)):
    """CPU-optimized image loader"""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def tensor_to_image(tensor):
    """Simplified CPU conversion"""
    return Image.fromarray(np.uint8(tensor * 255))
