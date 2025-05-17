import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
import time

class HighQualityStyleTransfer:
    def __init__(self):
        self.content_layers = ['block4_conv2']  # Changed layer selection
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.model = self._build_model()
        self.iterations = 300
        self.loss_calls = 0
        
    def _build_model(self):
        vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        layer_names = self.style_layers + self.content_layers
        outputs = [vgg.get_layer(name).output for name in layer_names]
        return Model(inputs=vgg.input, outputs=outputs)
        
    def gram_matrix(self, input_tensor):
        """Calculate style feature representation using Gram matrix"""
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations
        
    def transfer_style(self, content_image, style_image, style_weight=1e6, content_weight=1e0, 
                       progress_callback=None):
        # Convert to float32 numpy arrays
        content_array = content_image.astype(np.float32)
        style_array = style_image.astype(np.float32)
        
        # Process for VGG input
        content_vgg = tf.keras.applications.vgg19.preprocess_input(content_array * 255.0)
        style_vgg = tf.keras.applications.vgg19.preprocess_input(style_array * 255.0)
        
        # Expand to batch dimension
        content_batch = content_vgg[np.newaxis, ...]
        style_batch = style_vgg[np.newaxis, ...]
        
        # Extract style and content features
        style_features = self.model(style_batch)
        content_features = self.model(content_batch)
        
        # Compute style targets (Gram matrices)
        num_style_layers = len(self.style_layers)
        style_targets = [self.gram_matrix(style_features[i]) for i in range(num_style_layers)]
        
        # Get content targets
        content_targets = [content_features[num_style_layers]]
        
        # Initialize with the content image
        initial_image = content_array.copy()
        initial_image = tf.keras.applications.vgg19.preprocess_input(initial_image * 255.0)
        image_var = tf.Variable(initial_image)
        
        # For tracking progress
        start_time = time.time()
        best_loss = float('inf')
        best_img = None
        
        # Optimization function for L-BFGS
        def compute_loss(x):
            self.loss_calls += 1
            
            # Reshape the flattened vector to image dimensions
            current_image = x.reshape(content_array.shape)
            image_var.assign(current_image)
            
            with tf.GradientTape() as tape:
                # Get features of current image
                outputs = self.model(image_var[np.newaxis, ...])
                
                # Style loss
                style_output_features = outputs[:num_style_layers]
                style_score = 0
                for i in range(num_style_layers):
                    style_gram = self.gram_matrix(style_output_features[i])
                    style_score += tf.reduce_mean(tf.square(style_gram - style_targets[i]))
                style_score *= style_weight / num_style_layers
                
                # Content loss
                content_output_features = outputs[num_style_layers:]
                content_score = content_weight * tf.reduce_mean(
                    tf.square(content_output_features[0] - content_targets[0])
                )
                
                # Total loss
                loss = style_score + content_score
                
            # Get gradient
            grad = tape.gradient(loss, image_var)
            grad_flat = grad.numpy().flatten().astype(np.float64)
            loss_value = loss.numpy().astype(np.float64)
            
            # Progress reporting
            if self.loss_calls % 10 == 0:
                print(f"Iteration {self.loss_calls}: loss={loss_value:.2f}")
                if progress_callback:
                    progress = min(self.loss_calls / self.iterations, 0.95)
                    progress_callback(progress)
                
            return loss_value, grad_flat
        
        # Run L-BFGS optimization
        x_opt, f_val, _ = fmin_l_bfgs_b(
            func=compute_loss, 
            x0=image_var.numpy().flatten(), 
            maxfun=self.iterations, 
            maxiter=self.iterations,
            factr=1e7
        )
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
        
        # Process the result
        final_image = x_opt.reshape(content_array.shape)
        
        # Convert from VGG preprocessing back to displayable image
        final_image = self._deprocess_vgg(final_image)
        
        print(f"Style transfer completed in {time.time() - start_time:.1f}s")
        return final_image
        
    def _deprocess_vgg(self, processed_img):
        """Convert from VGG preprocessing back to displayable image"""
        img = processed_img.copy()
        
        # Undo VGG preprocessing
        img += np.array([103.939, 116.779, 123.68]).reshape((1, 1, 3))
        img = img[..., ::-1]  # BGR to RGB
        
        # Clip to valid range and return
        return np.clip(img, 0, 255) / 255.0
