## Overview

This project is an interactive web application for **Image Style Transfer**: the process of blending the content of one image (e.g., a cityscape) with the style of another (e.g., ocean waves or a painting), producing a new, stylized image. The app is built with **Streamlit** for the user interface and leverages **TensorFlow** and **Keras** for deep learning computations. The approach is based on the classic optimization-based neural style transfer method (Gatys et al., 2015), enhanced for usability and educational value.

---

## Features

- **Upload any content and style image** to generate a stylized blend.
- **Adjust style strength, content preservation, and number of iterations** for fine control.
- **Progress bar and live feedback** during processing.
- **Download the resulting stylized image**.
- **Runs entirely on CPU** for maximum compatibility.

---

## Technologies Used

- **Python 3.8+**
- [Streamlit](https://streamlit.io/) - Interactive web app framework
- [TensorFlow 2.x](https://www.tensorflow.org/) - Deep learning backend
- [Keras (VGG19)](https://keras.io/api/applications/vgg/) - Pre-trained convolutional neural network
- [NumPy](https://numpy.org/) - Numerical operations
- [SciPy (L-BFGS optimizer)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html) - Efficient optimization
- [Pillow](https://python-pillow.org/) - Image processing

---

## How It Works

1. **Feature Extraction**:  
   The app uses a pre-trained VGG19 network to extract feature representations from both the content and style images.

2. **Loss Calculation**:  
   - **Content Loss**: Measures the difference between the content features of the generated image and the original content image.
   - **Style Loss**: Uses Gram matrices to capture the style (textures and colors) of the style image and compares them to the generated image.
   - **Total Loss**: A weighted sum of content and style loss, controlled by user-adjustable sliders.

3. **Optimization**:  
   The generated image is initialized as a noisy version of the content image and iteratively updated using the L-BFGS optimizer to minimize the total loss.

4. **Result**:  
   After the optimization, the final image is post-processed and displayed for download.

---

## Installation

1. **Clone the Repository**

   git clone [https://github.com/yourusername/neural-style-transfer-app.git](https://github.com/preethammmm/Image-Style-Transfer)
   cd neural-style-transfer-app


2. **Install Dependencies**
  
   pip install -r requirements.txt


3. **Run the App**
   
   streamlit run app.py


---

## Usage

1. Open the app in your browser (Streamlit will provide a local URL).
2. Upload a **content image** (the subject you want to stylize).
3. Upload a **style image** (the artwork or texture you want to apply).
4. Adjust the **style strength**, **content preservation**, and **iterations** as desired.
5. Click **"Start Style Transfer"** and wait for the process to complete.
6. Download your stylized image!

---

- **Long Processing Time**: Optimization-based style transfer is slow, especially on CPU (can take several minutes for high-res images).
- **Parameter Sensitivity**: Balancing style and content weights is non-trivial and can lead to washed-out or unrecognizable results.
- **Gray or Blank Outputs**: Early bugs with image normalization, initialization, or loss calculation led to gray or incorrect outputs.
- **TensorFlow 2.x Migration**: Adapting classic style transfer code (written for TensorFlow 1.x) to eager execution and modern APIs required careful debugging.
- **Resource Usage**: Running on CPU avoids GPU compatibility issues but increases runtime.

---

## Room for Improvement

- **Speed**: Integrate fast style transfer models (e.g., AdaIN, Johnson et al.) for real-time results.
- **Quality**: Add total variation loss for smoother images and experiment with different layer combinations.
- **User Experience**: Allow real-time previews, batch processing, and video style transfer.
- **Model Options**: Support for transformer-based or attention-based style transfer models.
- **Evaluation**: Add metrics (e.g., SSIM, content/style loss) and a gallery of results.
- **Deployment**: Package as a Docker container or deploy to cloud platforms for broader accessibility.

---

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [TensorFlow Neural Style Transfer Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)

---
