from scipy.signal import convolve2d
import numpy as np
import copy
from imageio import imread
from skimage.color import rgb2gray
import scipy.ndimage


GRAY_REP = 1
THREE_DIMS = 3
TOTAL_GRAY_LEVELS = 256
MAX_GRAY_SHADE = 255

MIN_SIZE = 16


def read_image(filename, representation):
    '''
    The function returns normalized np.float64 image as requested.
    :param filename: the filename of an image on disk
    :param representation: 1 for gray-scale output, 2 for RGB image (only for rgb)
    :return: an image with the requested representation
    '''
    im = imread(filename)
    im_float = im.astype(np.float64)
    hist, bins = np.histogram(im, bins=TOTAL_GRAY_LEVELS)
    if len(np.where(hist > 1)[0]) > 0:
        im_float /= MAX_GRAY_SHADE
    if representation == GRAY_REP and im.ndim == THREE_DIMS:
        return rgb2gray(im_float)
    return im_float


def get_binomial_coefficients(filter_size):
    '''
    the function calculate and return a row vector with the binomial coefficients
    :param filter_size: the size of the vector
    :return: row vector with the binomial coefficients
    '''
    kernel = np.array([[1, 1]])
    fil = np.array([[1, 1]])
    for i in range(filter_size - 2):
        fil = scipy.signal.convolve2d(fil, kernel)
    return fil / np.sum(fil)


def reduce(img, filter_vec):
    '''
    the function sub-samples the give image in every second pixel in every second row
    :param img: the original image
    :param filter_vec: vector to use for bluring
    :return: new smaller image
    '''
    rows_blur = scipy.ndimage.filters.convolve(img, filter_vec)
    cols_blur = scipy.ndimage.filters.convolve(rows_blur, filter_vec.T)
    return cols_blur[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    the function construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter
    :return: pyr - the resulting grayscale pyramid as a standard python array
             filter_vec - row vector of shape (1, filter_size)
    """
    filter_vec = get_binomial_coefficients(filter_size)
    pyr = [copy.deepcopy(im)]

    for i in range(1, max_levels):
        reduced = reduce(pyr[i - 1], filter_vec)
        N, M = reduced.shape
        if N < MIN_SIZE or M < MIN_SIZE:
            break
        else:
            pyr.append(reduced)
    return pyr, filter_vec


''' --------------------------------------------------------- '''


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img