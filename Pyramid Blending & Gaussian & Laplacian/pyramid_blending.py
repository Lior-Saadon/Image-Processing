import numpy as np
import matplotlib.pyplot as plt
import copy
from imageio import imread
from skimage.color import rgb2gray
import os
import scipy.ndimage


GRAY_REP = 1
THREE_DIMS = 3
TOTAL_GRAY_LEVELS = 256
MAX_GRAY_SHADE = 255

MIN_SIZE = 16

EXAMPLE1_IM1 = "presubmit_externals/nba.jpg"
EXAMPLE1_IM2 = "presubmit_externals/pizza.jpg"
EXAMPLE1_MASK = "presubmit_externals/mask_pizza.jpg"

EXAMPLE2_IM1 = "presubmit_externals/airplane.jpg"
EXAMPLE2_IM2 = "presubmit_externals/tinkerbell.jpg"
EXAMPLE2_MASK = "presubmit_externals/tinkerbell_mask.jpg"


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


def expand(img, filter_vec):
    '''
    the function expands the give image using zero-padding between each neighbours pixels
    :param img: the original image
    :param filter_vec: vector to use for bluring
    :return: new bigger and padded image
    '''
    N, M = img.shape
    expanded = np.zeros((2*N, 2*M)).astype(img.dtype)
    expanded[::2, ::2] = img
    rows_blur = scipy.ndimage.filters.convolve(expanded, 2 * filter_vec)
    cols_blur = scipy.ndimage.filters.convolve(rows_blur, 2 * filter_vec.T)
    return cols_blur


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    the function construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter
    :return: pyr - the resulting grayscale pyramid as a standard python array
             filter_vec - row vector of shape (1, filter_size)
    """
    laplacian = []
    gaussian, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(gaussian) - 1):
        laplacian.append(gaussian[i] - expand(gaussian[i + 1], filter_vec))
    laplacian.append(gaussian[-1])
    return laplacian, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    reconstruction of an image from its laplacian pyramid
    :param lpyr: laplacian pyramid
    :param filter_vec: row vector of shape (1, filter_size)
    :param coeff: python list with length of max_levels in lpyr
    :return: reconstructed image
    '''
    pyr = np.multiply(lpyr, coeff)
    for i in range(len(coeff) - 1, 0, -1):
        pyr[i - 1] += (expand(pyr[i], filter_vec))
    return pyr[0]


def render_pyramid(pyr, levels):
    '''
    the func create and return united image of the pyramid levels
    :param pyr: laplacian or gaussian pyramid
    :param levels: total number of levels to present in the result
    :return: single black image in which the pyramid levels of the given
            pyramid are stacked horizontally
    '''
    min = np.min(pyr[0])
    max = np.max(pyr[0])
    img = pyr[0] - min / max
    N, M = pyr[0].shape
    for i in range(1, levels):
        Ni, Mi = np.shape(pyr[i])
        zeros = np.zeros((N - Ni, Mi))
        normalized = pyr[i] - min / max
        partial = np.concatenate((normalized, zeros), axis=0)
        img = np.concatenate((img, partial), axis=1)
    return img


def display_pyramid(pyr, levels):
    '''
    the func display united image of the pyramid levels
    :param pyr: laplacian or gaussian pyramid
    :param levels: total number of levels to present in the result
    '''
    plt.figure()
    plt.imshow(render_pyramid(pyr, levels), cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
    the function crete blended image.
    :param im1: grayscale image to blend
    :param im2: grayscale image to blend
    :param mask: boolean mask to indicate which parts of the images should appear in result
    :param max_levels: max number of levels to create pyramids
    :param filter_size_im: the size of the gaussian filter used to create laplacian pyramid
    :param filter_size_mask: the size of the gaussian filter used to create the gaussian pyramid of mask
    :return: im_blend - the blended image of im1 and im2
    '''
    L1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    mask = np.double(mask)
    Gm, filter3 = build_gaussian_pyramid(mask, max_levels, filter_size_mask)

    L_out = []
    for i in range(max_levels):
        L_out.append(Gm[i] * L1[i] + (1 - Gm[i]) * L2[i])
    im_blend = laplacian_to_image(L_out, filter3, np.ones(max_levels))
    return np.clip(im_blend, 0, 1)


def relpath(filename):
    '''
    :param filename: the file name
    :return: relative path name
    '''
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    '''
    the function create blended image of specific set of images
    :return: im1, im2, mask and the blended image
    '''
    return blend(EXAMPLE1_IM1, EXAMPLE1_IM2, EXAMPLE1_MASK, 5, 5, 5)


def blending_example2():
    '''
    the function create blended image of specific set of images
    :return: im1, im2, mask and the blended image
    '''
    return blend(EXAMPLE2_IM1, EXAMPLE2_IM2, EXAMPLE2_MASK, 4, 5, 5)


def blend(im1_filename, im2_filename, mask_filename, max_levels, filter_size_im, filter_size_mask):
    '''
    the function makes a blended image by each of the RGB channels separately,
    and display the result alongside the original images
    :param im1_filename: grayscale image filename to blend
    :param im2_filename: grayscale image filename to blend
    :param mask_filename: boolean mask filename to indicate which parts of the images should appear in result
    :param max_levels: max number of levels to create pyramids
    :param filter_size_im: the size of the gaussian filter used to create laplacian pyramid
    :param filter_size_mask: the size of the gaussian filter used to create the gaussian pyramid of mask
    :return: the blended image
    '''

    im1 = read_image(relpath(im1_filename), 2)
    im2 = read_image(relpath(im2_filename), 2)
    mask = read_image(relpath(mask_filename), 1).astype(np.bool)

    im_blend = np.zeros(im1.shape)
    for i in range(im1.shape[2]):
        im_blend[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask,
                                             max_levels, filter_size_im, filter_size_mask)
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(im1)
    axarr[0, 1].imshow(im2)
    axarr[1, 0].imshow(mask, cmap='gray')
    axarr[1, 1].imshow(im_blend)
    plt.show()

    return im1, im2, mask, im_blend



blending_example1()
blending_example2()