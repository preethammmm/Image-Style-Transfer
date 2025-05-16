import streamlit as st
import time
import io
from modules.high_quality import HighQualityStyleTransfer
from modules.utils import load_image, tensor_to_image

def show():
    st.set_page_config(page_title="🎨 CPU Style Transfer", layout="wide")
    st.title("CPU-Optimized Neural Style Transfer")
    
    with st.sidebar:
        st.header("Settings")
        iterations = st.slider("Iterations", 50, 300, 150)
        style_weight = st.slider("Style Strength", 1e3, 1e5, 1e4)
        content_weight = st.slider("Content Preservation", 1e-3, 1e-1, 1e-2)

    content_file = st.file_uploader("Content Image", type=["jpg", "png"])
    style_file = st.file_uploader("Style Image", type=["jpg", "png"])

    if content_file and style_file:
        if st.button("✨ Start Style Transfer"):
            start_time = time.time()
            
            # Load images
            content_img = load_image(content_file)
            style_img = load_image(style_file)
            
            # Initialize model
            model = HighQualityStyleTransfer()
            model.iterations = iterations
            
            # Process
            output = model.transfer_style(content_img, style_img, style_weight, content_weight)
            result_img = tensor_to_image(output)
            
            # Display
            st.image(result_img, use_container_width=True)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "stylized.png", "image/png")
            st.info(f"Processed in {time.time()-start_time:.1f}s on CPU")

if __name__ == "__main__":
    show()
