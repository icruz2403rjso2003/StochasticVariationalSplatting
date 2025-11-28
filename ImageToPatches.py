
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

class ImageToPatches:

    def __init__(self, image : np.ndarray, patch_size : int) -> None:
        
        self.image = image

        self.H, self.W = self.image.shape[0 : -1]

        self.patch_size = patch_size

        self.H_n_patches = np.ceil(self.H/patch_size).astype(int)

        self.W_n_patches = np.ceil(self.W/patch_size).astype(int)

        self.pixel = np.zeros(shape = (self.H*self.W, 2))

        self.color = np.zeros(shape = (self.H*self.W, 3))

        self.pixel_patches = np.zeros(shape = (self.H_n_patches*self.patch_size, self.W_n_patches*self.patch_size, 2))

        self.color_patches = np.zeros(shape = (self.H_n_patches*self.patch_size, self.W_n_patches*self.patch_size, 3))

    def image_to_pixel(self) -> None:

        self.pixel = np.indices((self.H, self.W)).reshape((2, self.H*self.W)).T

        self.pixel = self.pixel/self.pixel.max(axis = 0)

    def pixel_to_patches(self) -> None:

        self.pixel_patches[ : self.H, : self.W] = self.pixel.reshape(self.H, self.W, 2)

        self.pixel_patches = sliding_window_view(self.pixel_patches, (self.patch_size, self.patch_size, 2))

        self.pixel_patches = self.pixel_patches[ : : self.patch_size, : : self.patch_size, 0, 0]

    def image_to_color(self) -> None:

        self.color = self.image.reshape((self.H*self.W, 3))

        self.color = self.color/self.color.max(axis = 0)

    def color_to_patches(self) -> None:

        self.color_patches[ : self.H, : self.W] = self.color.reshape(self.H, self.W, 3)

        self.color_patches = sliding_window_view(self.color_patches, (self.patch_size, self.patch_size, 3))

        self.color_patches = self.color_patches[ : : self.patch_size, : : self.patch_size, 0, 0]

    def image_to_patches(self) -> None:

        self.image_to_pixel()

        self.pixel_to_patches()

        self.image_to_color()

        self.color_to_patches()
