import tensorflow as tf
import tensorflow_hub as hub

class FastStyleTransfer:
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    
    def transfer_style(self, content_image, style_image):
        # Convert numpy arrays to TensorFlow tensors explicitly
        content_tensor = tf.constant(content_image, dtype=tf.float32)
        style_tensor = tf.constant(style_image, dtype=tf.float32)
        return self.model(content_tensor, style_tensor)[0].numpy()
