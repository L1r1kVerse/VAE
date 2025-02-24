# Variational Autoencoder (VAE)
This project explores and implements ideas proposed in the landmark deep learning paper "Auto-Encoding Variational Bayes".

## Project Structure
- **`VAE.ipynb`**: The main notebook, containing exploration of the main ideas of the paper.

- **`VAE_implementation.ipynb`**: A notebook implementing first an Autoencoder and then a Variational Autoencoder using the ideas from the paper.

- **`Secondary files:`**:
    - **`helper_functions.py:`**: Helper functions for the project.
  

## Results

### Autoencoder
<p align="center">
  <img src="data/model_architecturev2.png" width="1200"/>
</p>
Reconstructions from an Autoencoder - only trained with reconstruction loss:
<p align="center">
  <img src="data/autoenc_reconstructions.png" width="800"/>
</p>
Attempting to generate new samples with the Autoencoder:
<p align="center">
  <img src="data/autoenc_samples.png" width="800"/>
</p>

### Variational Autoencoder(VAE)
<p align="center">
  <img src="data/vae.png" width="1200"/>
</p>
Reconstructions from a VAE - trained with reconstruction + KL Divergence loss:
<p align="center">
  <img src="data/vae_reconstructions.png" width="800"/>
</p>
Generating new samples with the VAE:
<p align="center">
  <img src="data/vae_samples.png" width="800"/>
</p>
Impact of varying only one dimension of the latent vector on generated samples:
<p align="center">
  <img src="data/vae_varying_one_latent_dim.png" width="800"/>
</p>

Visualizing an area of the latent space for a VAE with 2D latent:
<p align="center">
  <img src="data/vae_2d_latent_space.png" width="1200"/>
</p>


### Paper and Blog References:

#### Papers:
- [x]  [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) <br>
- [x] [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)<br>
- [x] [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)

#### Blogs:
- &nbsp; [The Reparameterization Trick](https://gregorygundersen.com/blog/2018/04/29/reparameterization/)
- &nbsp; [VAE from scracth](https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95)







