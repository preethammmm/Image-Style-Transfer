import streamlit as st
import time
import io
import numpy as np
from modules.high_quality import HighQualityStyleTransfer
from modules.utils import load_image, tensor_to_image

def show():
    st.set_page_config(page_title="🎨 Neural Style Transfer", layout="wide")
    st.title("Artistic Style Transfer App")

    # ========== Sidebar Controls ==========
    with st.sidebar:
        st.header("Settings")
        iterations = st.slider("Iterations", 50, 500, 150, 
                             help="More iterations = better quality but slower processing")
        style_weight = st.slider("Style Strength", 1e5, 1e7, 1e6)
        content_weight = st.slider("Content Preservation", 1e-2, 1e2, 1e0)
        
        st.markdown("---")
        st.info("ℹ️ For best results:\n- Use high-contrast style images\n- Start with 150 iterations\n- Balance style/content weights")

    # ========== Main UI ==========
    col1, col2 = st.columns(2)
    with col1:
        content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    with col2:
        style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    if content_file and style_file:
        try:
            # ========== Initialize Processing ==========
            if st.button("✨ Start Style Transfer", type="primary"):
                start_time = time.time()
                progress_bar = st.progress(0)
                status_text = st.empty()

                # ========== Image Preprocessing ==========
                status_text.info("🖼️ Loading and resizing images...")
                content_img = load_image(content_file)
                style_img = load_image(style_file)
                progress_bar.progress(10)

                # ========== Style Transfer ==========
                status_text.info("🎨 Applying style transfer...")
                model = HighQualityStyleTransfer()
                model.iterations = iterations
                
                # Convert weights to scientific notation
                style_weight_sci = style_weight * 1e-4  # Map 10,000 → 1e4
                content_weight_sci = content_weight * 1e-3  # Map 10 → 1e-2
                
                # Process with progress updates
                def update_progress(progress):
                    current_progress = int(10 + progress * 85)
                    progress_bar.progress(current_progress)
                    status_text.info(f"⏳ Processing: {current_progress}% complete")
                
                output = model.transfer_style(
                    content_img, style_img,
                    style_weight=style_weight_sci,
                    content_weight=content_weight_sci,
                    progress_callback=update_progress
                )
                
                # ========== Display Results ==========
                progress_bar.progress(95)
                status_text.info("📸 Finalizing output...")
                
                result_img = tensor_to_image(output)
                st.image(result_img, use_container_width=True)
                progress_bar.progress(100)
                
                # Download button
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                st.download_button(
                    "💾 Download Stylized Image",
                    buf.getvalue(),
                    file_name="stylized.png",
                    mime="image/png"
                )
                
                # Performance stats
                process_time = time.time() - start_time
                status_text.success(f"✅ Completed in {process_time:.1f} seconds!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

if __name__ == "__main__":
    show()
