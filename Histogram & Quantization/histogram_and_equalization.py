import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

GRAY_REP = 1
TWO_DIMS = 2
THREE_DIMS = 3
TOTAL_GRAY_LEVELS = 256
MAX_GRAY_SHADE = 255


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


def imdisplay(filename, representation):
    '''
    The function displays an image in a given representation
    :param filename: the filename of an image on disk
    :param representation: 1 for gray-scale output, 2 for RGB image (only for rgb)
    :return: none
    '''

    img = read_image(filename, representation)
    plt.figure()
    if img.ndim == TWO_DIMS:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.show()


def rgb2yiq(imRGB):
    '''
    The function convert RGB image to YIQ image
    :param imRGB: 3-dim np.float64 image
    :return: 3-dim np.float64 YIQ image
    '''
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    return np.dot(imRGB, mat.T)


def yiq2rgb(imYIQ):
    '''
    The function convert YIQ image to RGB image
    :param imYIQ: 3-dim np.float64 image
    :return: 3-dim np.float64 RGB image
    '''
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    return np.dot(imYIQ, np.linalg.inv(mat.T))


def hist(im_orig):
    '''
    The func receive 2D image and apply the histogram equalization algorithm
    :param im_orig: 2D image
    :return: a list [im_eq, hist_orig, hist_eq]
    '''

    hist_orig, bin_edges = np.histogram(im_orig, bins=TOTAL_GRAY_LEVELS)
    cumulative = np.cumsum(hist_orig)

    if min(cumulative) != 0 or max(cumulative) != MAX_GRAY_SHADE:
        cumulative = MAX_GRAY_SHADE * cumulative / cumulative[-1]  # normalized

    m = np.nonzero(cumulative)[0][0]

    lut = cumulative
    lut = lut - cumulative[m]
    lut = lut / (cumulative[MAX_GRAY_SHADE] - cumulative[m])
    lut = lut * MAX_GRAY_SHADE

    lut = np.rint(lut)

    shape = im_orig.shape

    im_orig = im_orig * MAX_GRAY_SHADE
    im_orig = np.rint(im_orig).astype(np.int)

    im_eq = lut[im_orig]
    im_eq = np.reshape(im_eq, shape)
    np.clip(im_eq, 0, 1)

    hist_eq, bin_edge = np.histogram(im_orig, bins=TOTAL_GRAY_LEVELS)

    return [im_eq, hist_orig, hist_eq]


def histogram_equalize(im_orig):
    '''
    The function performs histogram equalization to ao given image
    :param im_orig: the input grayscale\RGB float64 image with [0,1] values
    :return: a list [im_eq, hist_orig, hist_eq]
    '''

    if np.ndim(im_orig) == THREE_DIMS:           # RGB img
        temp = rgb2yiq(im_orig)
        y_img = temp[:, :, 0]
        im_eq, hist_orig, hist_eq = hist(y_img)
        temp[:, :, 0] = (im_eq / MAX_GRAY_SHADE).astype(np.float64)
        im_eq = yiq2rgb(temp)
        return [im_eq, hist_orig, hist_eq]
    else:                                       # grayscale img
        return hist(im_orig)


def quant(im_orig, n_quant, n_iter):
    """
    The function perform quantization on a given 2D [0,1] image
    :param im_orig: the input 2D image with [0,1] values
    :param n_quant: the number of intensities im_quant should have
    :param n_iter: the maximum number of iterations of the optimization procedure
    :return: a list of [im_quant ([0,1] image], error]
    """
    z = np.zeros(n_quant + 1,)
    q = np.zeros(n_quant)
    error = []

    hist_orig, bin_edges = np.histogram(im_orig, bins=TOTAL_GRAY_LEVELS)
    cumulative = np.cumsum(hist_orig)

    # initialization
    interval = cumulative[-1] / n_quant     # num of pixels for each interval
    for i in range(1, n_quant):
        # return array of indices with shape of (array([..,..,..], dtype=int32)
        z[i] = np.where(i * interval < cumulative)[0][0] - 1
    z[n_quant] = MAX_GRAY_SHADE
    z = np.rint(z).astype(np.int64)

    for iter in range(n_iter):
        # calculate q's
        for i in range(n_quant):             # iterate over z's
            mone = 0
            mecane = 0
            for j in range(z[i], z[i + 1]):
                mone += j * hist_orig[j]
                mecane += hist_orig[j]
            q[i] = mone / mecane

        # calculate z's
        new_z = np.zeros(n_quant + 1, )
        for i in range(1, n_quant):
            new_z[i] = (q[i - 1] + q[i]) / 2
        new_z[-1] = MAX_GRAY_SHADE
        new_z = np.ceil(new_z).astype(np.int64)

        # calculate error
        err = 0
        for i in range(n_quant):
            for j in range(z[i], z[i + 1]):
                err += pow(int((q[i]) - j), 2) * hist_orig[j]
        error.append(err)

        # check for convergence
        if np.array_equal(z, new_z):
            break
        z = new_z

    # build lookup table
    lut = np.zeros(TOTAL_GRAY_LEVELS)
    z = np.ceil(z).astype(np.int)
    for i in range(n_quant):
        lut[z[i]:z[i + 1]] = q[i]
    lut = np.round(lut)


    # apply lookup table
    im_orig = im_orig * MAX_GRAY_SHADE
    im_orig = np.rint(im_orig).astype(np.int)
    im_quant = lut[im_orig]
    im_quant /= MAX_GRAY_SHADE

    return [im_quant, error]


def quantize(im_orig, n_quant, n_iter):
    """
    The function perform quantization on a given image
    :param im_orig: the input grayscale\RGB float64 image with [0,1] values
    :param n_quant: the number of intensities im_quant should have
    :param n_iter: the maximum number of iterations of the optimization procedure
    :return: a list of [im_quant, error]
    """
    if np.ndim(im_orig) == THREE_DIMS:           # RGB img
        temp = rgb2yiq(im_orig)
        y_img = temp[:, :, 0]
        im_quant, error = quant(y_img, n_quant, n_iter)
        temp[:, :, 0] = (im_quant).astype(np.float64)
        im_quant = yiq2rgb(temp)
        return [im_quant, error]
    else:                                       # grayscale img
        return quant(im_orig, n_quant, n_iter)








