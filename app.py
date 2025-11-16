"""
MNIST Digit Generator - Interactive Web Application.

This Streamlit app provides an interactive interface to generate handwritten digits
using a pre-trained Conditional Variational Autoencoder (CVAE).

Author: Mansi Dhanania
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from config import DEVICE, LATENT_DIM, MAX_SAMPLES, MODEL_PATH, NUM_CLASSES
from cvae_mnist import CVAE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Page Configuration =============
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============= Custom CSS Styling =============
st.markdown("""
<style>
    /* Reduce top padding */
    .appViewContainer {
        padding-top: 0.5rem;
    }
    
    /* Compact title */
    h1 {
        margin-bottom: 0.25rem !important;
        padding-top: 0 !important;
    }
    
    /* Compact subtitle */
    .subtitle {
        margin-bottom: 0.5rem !important;
        font-size: 0.95rem !important;
    }
    
    /* Reduce markdown section spacing */
    h3 {
        margin-top: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact control inputs */
    .stSelectbox, .stRadio, .stSlider {
        margin-bottom: 0.25rem !important;
    }
    
    /* Reduce divider spacing */
    hr {
        margin: 0.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model() -> CVAE:
    """
    Load pre-trained CVAE model (cached for efficiency).

    Returns:
        Loaded CVAE model on appropriate device.

    Raises:
        FileNotFoundError: If model file not found.
        RuntimeError: If model loading fails.
    """
    try:
        model = CVAE(latent_dim=LATENT_DIM).to(DEVICE)
        model.load_state_dict(
            torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
        )
        model.eval()
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(
            f"Pre-trained model not found at {MODEL_PATH}. "
            "Please train the model first by running: python cvae_mnist.py"
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")


def generate_images(
    model: CVAE, digit: int, num_samples: int = MAX_SAMPLES, seed: int = None
) -> np.ndarray:
    """
    Generate digit images using the CVAE model.

    Args:
        model: CVAE model instance.
        digit: Digit to generate (0-9).
        num_samples: Number of images to generate.
        seed: Random seed for reproducibility (None for random).

    Returns:
        Array of generated images with shape (num_samples, 28, 28).

    Raises:
        ValueError: If digit not in range 0-9.
    """
    if not 0 <= digit <= 9:
        raise ValueError(f"Digit must be between 0 and 9, got {digit}")

    try:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        with torch.no_grad():
            # Create one-hot encoded labels
            labels = torch.full((num_samples,), digit, dtype=torch.long)
            y = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(DEVICE)

            # Sample from latent space
            z = torch.randn(num_samples, LATENT_DIM).to(DEVICE)

            # Generate images
            generated = model.decode(z, y).cpu().numpy()

        # Reshape to 28x28
        generated = generated.reshape(-1, 28, 28)
        logger.info(f"Generated {num_samples} images for digit {digit}")
        return generated

    except Exception as e:
        logger.error(f"Error generating images: {e}")
        raise RuntimeError(f"Failed to generate images: {e}")


def plot_images(images: np.ndarray, digit: int, cols: int = 5) -> plt.Figure:
    """
    Plot generated images in a professional grid layout.

    Args:
        images: Array of generated images with shape (num_samples, 28, 28).
        digit: Digit label for title.
        cols: Number of columns in the grid.

    Returns:
        Matplotlib figure object.
    """
    num_images = images.shape[0]
    rows = max(1, (num_images + cols - 1) // cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(12, 1.8 * rows), facecolor="white"
    )

    # Handle single subplot case
    if num_images == 1:
        axes = np.array([axes]).reshape(1, 1)
    elif rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < num_images:
            # Display image with high quality
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Sample {i+1}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.suptitle(
        f"Generated Digit: {digit}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main() -> None:
    """Main Streamlit application."""
    # Header - Ultra compact
    st.markdown(
        "<h1 style='text-align: center; color: #2C3E50; margin: 0; padding: 0; font-size: 28px;'>🔢 MNIST Digit Generator</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align: center; color: #7F8C8D; font-size: 15px; margin: 0.1rem 0 0.5rem 0; font-weight: 500;'>Generate handwritten digits using a Conditional VAE</p>",
        unsafe_allow_html=True,
    )

    # Main content
    try:
        # Load model
        with st.spinner("Loading model..."):
            model = load_model()

        # Control Panel - Compact
        st.markdown("<h4 style='margin: 0.2rem 0 0.3rem 0; font-size: 14px;'>🎮 Controls</h4>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="small")
        
        with col1:
            st.markdown("<p style='font-size: 0.9rem; margin: 0;'><b>Digit</b></p>", unsafe_allow_html=True)
            digit_options = ["Random"] + [str(i) for i in range(10)]
            selected = st.selectbox(
                "Select digit",
                options=digit_options,
                index=0,
                label_visibility="collapsed",
                key="digit_select",
            )
            
            if selected == "Random":
                digit = np.random.randint(0, 10)
            else:
                digit = int(selected)

        with col2:
            st.markdown("<p style='font-size: 0.9rem; margin: 0;'><b>Samples</b></p>", unsafe_allow_html=True)
            num_samples = st.selectbox(
                "Samples",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                index=MAX_SAMPLES - 1,
                label_visibility="collapsed",
                key="samples_select",
            )

        with col3:
            st.markdown("<p style='font-size: 0.9rem; margin: 0;'><b>Seed</b></p>", unsafe_allow_html=True)
            seed_col1, seed_col2 = st.columns([1, 2], gap="small")
            
            with seed_col1:
                seed_option = st.radio(
                    "Choose seed mode",
                    options=["Random", "Fixed"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="seed_radio",
                )
            
            with seed_col2:
                if seed_option == "Fixed":
                    seed = st.slider(
                        "Seed value",
                        min_value=0,
                        max_value=100,
                        value=42,
                        step=1,
                        label_visibility="collapsed",
                        key="seed_slider",
                    )
                else:
                    seed = None

        # Generate button - Compact
        col1, col2, col3 = st.columns([1, 2, 1], gap="small")
        with col2:
            generate_button = st.button(
                "Generate",
                use_container_width=True,
                type="primary",
                key="gen_btn",
            )

        # Generate and display
        if generate_button:
            try:
                with st.spinner("Generating..."):
                    images = generate_images(model, digit, num_samples, seed)
                
                st.success(f"✓ Generated {num_samples} variation(s)")
                
                # Show generated images
                fig = plot_images(images, digit)
                st.pyplot(fig, use_container_width=True)

            except (ValueError, RuntimeError) as e:
                st.error(f"Error: {e}")

        else:
            # Placeholder message
            st.info("Select a digit and click 'Generate' to create variations")

        # Model Information Section - Using Expander to save space
        with st.expander("📚 Learn More", expanded=False):
            st.markdown("""
            **Conditional Variational Autoencoder (CVAE)**
            
            This model is trained on the MNIST dataset to generate realistic handwritten digits. 
            Unlike traditional autoencoders, a CVAE learns a continuous latent space, allowing it 
            to generate novel digit variations.
            
            **Model Architecture:**
            - **Encoder**: Compresses 28×28 images → 20D latent vector
            - **Latent Space**: 20-dimensional Gaussian distribution
            - **Decoder**: Expands latent vector → reconstructed 28×28 images
            
            **How It Works:**
            1. Your selected digit (0-9) is combined with an image and passed through the encoder
            2. The model samples random points from the latent space
            3. The decoder reconstructs a new digit variation from the sampled point
            
            **Training Details:**
            - Dataset: MNIST (60,000 images)
            - Loss: Reconstruction + KL Divergence
            - Optimizer: Adam (learning rate: 0.001)
            - Parameters: ~665K
            """)

    except FileNotFoundError as e:
        st.error(
            f"""
            **Model Not Found**

            {str(e)}

            **To train the model:**
            ```bash
            python cvae_mnist.py
            ```
            """
        )
        st.stop()

    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)

    # Footer
    st.markdown(
        "<p style='text-align: center; color: #999; font-size: 11px; margin-top: 1rem;'><a href='https://github.com/MansiDhanania/digit-generator' target='_blank' style='color: #999; text-decoration: none;'>GitHub</a></p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
