import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Age Classification using ViT",
    page_icon="👤",
    layout="wide"
)

# App title
st.title("👤 Age Classification using Vision Transformer (ViT)")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a Vision Transformer (ViT) model to classify age ranges from images.
    
    **Model:** `nateraw/vit-age-classifier`
    
    **How to use:**
    1. Upload an image or enter an image URL
    2. Click 'Classify Age'
    3. View the predictions
    
    The model can classify into these age ranges:
    - 0-2
    - 3-9
    - 10-19
    - 20-29
    - 30-39
    - 40-49
    - 50-59
    - 60-69
    - 70+
    """)
    
    st.divider()
    st.markdown("Built with 🤗 Transformers and Streamlit")

# Initialize the model with caching
@st.cache_resource
def load_model():
    """Load the age classification model once and cache it"""
    try:
        classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
with st.spinner("Loading age classification model..."):
    age_classifier = load_model()

if age_classifier is None:
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Input Image")
    
    # Choose input method
    input_method = st.radio(
        "Select input method:",
        ["Upload an image", "Use image URL"]
    )
    
    image = None
    image_source = None
    
    if input_method == "Upload an image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                image_source = uploaded_file.name
                st.success(f"✅ Image loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error opening image: {e}")
    
    else:  # Use image URL
        url = st.text_input("Enter image URL:")
        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image_source = url
                st.success("✅ Image loaded from URL")
            except Exception as e:
                st.error(f"Error loading image from URL: {e}")
    
    # Display the uploaded image
    if image:
        st.image(image, caption=f"Input Image: {image_source}", use_column_width=True)

with col2:
    st.header("🔍 Predictions")
    
    # Add a classify button
    classify_button = st.button(
        "🎯 Classify Age",
        type="primary",
        disabled=image is None,
        use_container_width=True
    )
    
    if classify_button and image:
        with st.spinner("Analyzing image..."):
            try:
                # Make predictions
                age_predictions = age_classifier(image)
                
                # Sort predictions by score
                sorted_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
                
                # Display results
                st.subheader("📊 Age Classification Results")
                
                # Top prediction with emphasis
                top_prediction = sorted_predictions[0]
                st.metric(
                    label="**Most Likely Age Range**",
                    value=top_prediction['label'],
                    delta=f"{top_prediction['score']:.1%} confidence"
                )
                
                st.divider()
                
                # All predictions in a bar chart format
                st.subheader("All Predictions")
                
                # Create a DataFrame for visualization
                import pandas as pd
                df = pd.DataFrame(sorted_predictions)
                df['score_percent'] = df['score'] * 100
                
                # Display as bar chart
                st.bar_chart(df.set_index('label')['score_percent'])
                
                # Display as table
                st.dataframe(
                    df[['label', 'score']].rename(
                        columns={'label': 'Age Range', 'score': 'Confidence'}
                    ).style.format({'Confidence': '{:.2%}'}),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed breakdown
                with st.expander("📋 View Detailed Breakdown"):
                    for i, pred in enumerate(sorted_predictions):
                        col_pred, col_score = st.columns([3, 1])
                        with col_pred:
                            st.write(f"**{pred['label']}**")
                        with col_score:
                            st.write(f"{pred['score']:.2%}")
                        
                        # Add a progress bar for each prediction
                        st.progress(float(pred['score']))
                
            except Exception as e:
                st.error(f"Error during classification: {e}")
    
    elif image is None:
        st.info("👈 Please upload an image or enter a URL to get started")

# Footer
st.divider()
st.caption("Age Classification App | Powered by Hugging Face Transformers and Streamlit")
