from skimage.morphology import binary_closing, binary_erosion, square
import numpy as np

def postpro(mask):
	selem = square(5)
	return binary_erosion(binary_closing(mask, selem), selem)