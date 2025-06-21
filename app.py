# Import Necessary Libraries
import torch
import streamlit as st
import numpy as np
from cvae_mnist import CVAE
import matplotlib.pyplot as plt

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVAE(latent_dim=20).to(device)
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

st.title("MNIST Digit Generator")
digit = st.selectbox("Select a digit to generate", list(range(10)))

# Generate 5 samples
if st.button("Generate Images"):
    y = torch.nn.functional.one_hot(torch.tensor([digit]*5), 10).float().to(device)
    z = torch.randn(5, 20).to(device)
    with torch.no_grad():
        out = model.decode(z, y).cpu().numpy()
    out = out.reshape(-1, 28, 28)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(out[i], cmap="gray")
        ax.axis("off")
    st.pyplot(fig)