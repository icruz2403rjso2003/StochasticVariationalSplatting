
import numpy as np

import jax.numpy as jnp

class ImageToPatches:

    def __init__(self, image : np.ndarray, patch_size : int) -> None:
        
        self.image = jnp.asarray(image)

        self.H, self.W = self.image.shape[0 : -1]

        self.patch_size = patch_size

        self.H_n_patches = jnp.ceil(self.H/patch_size).astype(int)

        self.W_n_patches = jnp.ceil(self.W/patch_size).astype(int)

        self.H_pad = self.H_n_patches*self.patch_size

        self.W_pad = self.W_n_patches*self.patch_size

        self.pixel = jnp.zeros(shape = (self.H*self.W, 2))

        self.color = jnp.zeros(shape = (self.H*self.W, 3))

        self.pixel_patches = jnp.zeros(shape = (self.H_n_patches, self.W_n_patches, self.patch_size*self.patch_size, 2))

        self.color_patches = jnp.zeros(shape = (self.H_n_patches, self.W_n_patches, self.patch_size*self.patch_size, 3))

    def image_to_pixel(self) -> None:

        self.pixel = jnp.indices((self.H, self.W)).reshape((2, self.H*self.W)).T

        self.pixel = self.pixel/self.pixel.max(axis = 0)

    def pixel_to_patches(self) -> None:

        padding = jnp.zeros(shape = (self.H_pad, self.W_pad, 2))

        padding = padding.at[ : self.H, : self.W].set(self.pixel.reshape(self.H, self.W, 2))

        padding = padding.reshape((self.H_pad*self.W_pad, 2))

        self.pixel_patches = padding.reshape((self.H_n_patches, self.W_n_patches, self.patch_size*self.patch_size, 2))

    def image_to_color(self) -> None:

        self.color = self.image.reshape((self.H*self.W, 3))

        self.color = self.color/self.color.max(axis = 0)

    def color_to_patches(self) -> None:

        padding = jnp.zeros(shape = (self.H_pad, self.W_pad, 3))

        padding = padding.at[ : self.H, : self.W].set(self.color.reshape(self.H, self.W, 3))

        padding = padding.reshape((self.H_pad*self.W_pad, 3))

        self.color_patches = padding.reshape((self.H_n_patches, self.W_n_patches, self.patch_size*self.patch_size, 3))

    def image_to_patches(self) -> None:

        self.image_to_pixel()

        self.pixel_to_patches()

        self.image_to_color()

        self.color_to_patches()
