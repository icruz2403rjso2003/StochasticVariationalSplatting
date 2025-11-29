
import jax

import jax.numpy as jnp

def image_to_patches(image : jnp.ndarray, patch_size : int) -> tuple:

    H, W = image.shape[0 : -1]

    H_n_patches = (H + patch_size - 1)//patch_size

    W_n_patches = (W + patch_size - 1)//patch_size

    H_pad = H_n_patches*patch_size

    W_pad = W_n_patches*patch_size

    yy, xx = jnp.meshgrid(jnp.linspace(0, 1, H), jnp.linspace(0, 1, W), indexing = 'ij')

    pixel = jnp.stack([yy, xx], axis = -1)

    color = image/image.max(axis = (0, 1))

    pad_spec = ((0, H_pad - H), (0, W_pad - W), (0, 0))

    pixel_pad = jnp.pad(pixel, pad_spec, mode = 'constant')

    color_pad = jnp.pad(color, pad_spec, mode = 'constant')

    def pad_to_patches(pad) -> jnp.array:

        pad = pad.reshape(H_n_patches, patch_size, W_n_patches, patch_size, -1)

        pad = pad.swapaxes(1, 2)
        
        return pad.reshape(H_n_patches, W_n_patches, patch_size*patch_size, pad.shape[-1])
        
    pixel_patches = pad_to_patches(pad = pixel_pad)

    color_patches = pad_to_patches(pad = color_pad)

    return pixel_patches, color_patches

image_to_patches = jax.jit(image_to_patches, static_argnames = ('patch_size', ))