import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from scipy.optimize import fmin_l_bfgs_b

# Force CPU-only execution
tf.config.set_visible_devices([], 'GPU')

class HighQualityStyleTransfer:
    def __init__(self):
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.model = self._build_model()
        self.iterations = 150  # Reduced for CPU

    def _build_model(self):
        vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in self.style_layers + self.content_layers]
        return Model(vgg.input, outputs)

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        return result / tf.cast(input_shape[1]*input_shape[2], tf.float32)

    def transfer_style(self, content_image, style_image, style_weight=1e4, content_weight=1e-2):
        # Precompute targets
        style_outputs = self.model(style_image[np.newaxis, ...])
        style_targets = [self.gram_matrix(feat) for feat in style_outputs[:5]]
        
        content_outputs = self.model(content_image[np.newaxis, ...])
        content_targets = content_outputs[5:]

        # L-BFGS optimization with GradientTape
        def loss_and_grads(x):
            x = x.reshape((1,) + content_image.shape)
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(x)
                outputs = self.model(x)
                
                # Calculate losses
                style_loss = 0
                for i in range(5):
                    style_loss += tf.reduce_mean((self.gram_matrix(outputs[i]) - style_targets[i])**2)
                
                content_loss = tf.reduce_mean((outputs[5] - content_targets[0])**2)
                total_loss = style_weight*style_loss + content_weight*content_loss
                
            grads = tape.gradient(total_loss, x)
            return total_loss.numpy().astype('float64'), grads.numpy().flatten().astype('float64')

        # Optimize with L-BFGS
        result, _, _ = fmin_l_bfgs_b(
            func=loss_and_grads,
            x0=content_image.flatten(),
            maxfun=self.iterations,
            factr=10
        )
        return result.reshape(content_image.shape)
