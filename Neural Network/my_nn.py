import tensorflow as tf
import numpy as np
import random
import copy
import utils
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from imageio import imread, imsave
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve

GRAY_REP = 1
THREE_DIMS = 3
TOTAL_GRAY_LEVELS = 256
MAX_GRAY_SHADE = 255

FAILURE = -1
SUB = 0.5

KERNEL_SIZE = 3
CONV_KERNEL = (3, 3)

TRAIN_PRCNT = 0.8
ADAM_BETA = 0.9

MIN_SIGMA = 0
MAX_SIGMA = 0.2

LIST_KER_SIZE = [7]

noising_corruption_function = lambda image : add_gaussian_noise(image, MIN_SIGMA, MAX_SIGMA)
blurring_corruption_function = lambda image : random_motion_blur(image, LIST_KER_SIZE)



def read_image(filename, representation):
    """
    The function returns normalized np.float64 image as requested.
    :param filename: the filename of an image on disk
    :param representation: 1 for gray-scale output, 2 for RGB image (only for rgb)
    :return: an image with the requested representation
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    hist, bins = np.histogram(im, bins=TOTAL_GRAY_LEVELS)
    if len(np.where(hist > 1)[0]) > 0:
        im_float /= MAX_GRAY_SHADE
    if representation == GRAY_REP and im.ndim == THREE_DIMS:
        return rgb2gray(im_float)
    return im_float


def crop_im(im, x, y, height, width):
    '''
    helper function to load_dataset
    :param im: image
    :param x, y: coordinates
    :param height, width: integers
    :return: cropped image
    '''
    return im[y:y + height, x:x + width]


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator which yields batches of tuples- original and currupted image patches.
    :param filenames: a list of filenames of clean images
    :param batch_size: the size of the batch of images for each iteratio of SGD
    :param corruption_func: a function receiving a numpy's array representation of an image as a single arg, and return a randomly cuttupted version of the input image
    :param crop_size: a tuple (height, width) specify the crop size of the batches to extract
    :return: a generator which yield each time a set of corrupted and original image patches
    """
    cache = {}
    height, width = crop_size

    while True:
        target_batch = np.empty((batch_size, height, width, 1))   # clean
        source_batch = np.empty((batch_size, height, width, 1))   # currupted

        for i in range(batch_size):
            filename = np.random.choice(filenames)

            if filename in cache:
                im = cache[filename]
            else:
                im = read_image(filename, GRAY_REP)
                cache[filename] = im

            y, x = np.random.randint(im.shape[0] - 3*height), np.random.randint(im.shape[1] - 3*width)  # choose top left corner of big patch to corrupt
            big_crop = crop_im(im, x, y, 3*height, 3*width)
            corrupted = corruption_func(crop_im(im, x, y, 3*height, 3*width))  # corrupt big patch

            y, x = np.random.randint(big_crop.shape[0] - height), np.random.randint(big_crop.shape[1] - width)  # choose top left corner of sliced patch
            corp_patch = corrupted[y:y + height, x:x + width]
            orig_patch = big_crop[y:y + height, x:x + width]

            target_batch[i, :, :, 0] = orig_patch - SUB
            source_batch[i, :, :, 0] = corp_patch - SUB

        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    The function takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration
    :param input_tensor: symbolic input tensor
    :param num_channels: number of channels for wach of the convolutional layers
    :return: the symbolic output tensor of the layer configuration
    """
    conv1 = Conv2D(num_channels, CONV_KERNEL, padding='same')(input_tensor)
    actv = Activation('relu')(conv1)
    conv2 = Conv2D(num_channels, CONV_KERNEL, padding='same')(actv)
    combined = Add()([input_tensor, conv2])
    output_tensor = Activation('relu')(combined)
    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    the function return the complete neural network model
    :param height: input height
    :param width: input width
    :param num_channels: number of output channels in the network (except the last one which has 1)
    :param num_res_blocks: number of residual blocks
    :return: untrained keras model
    """
    input_tensor = Input(shape=(height, width, 1))
    conv = Conv2D(num_channels, CONV_KERNEL, padding ='same')(input_tensor)

    block = conv
    for i in range(num_res_blocks):
        block = resblock(block, num_channels)

    output = Conv2D(1, CONV_KERNEL, padding='same')(block)
    combined = Add()([output, input_tensor])
    model = Model(inputs=input_tensor, outputs=combined)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    the function trains a neural network
    :param model: a general neural network model for image restoration
    :param images: a list of file paths pointing to image files.
    :param corruption_func: corruption function
    :param batch_size: the size of batch of examples for each iteration of SGD
    :param steps_per_epoch: number of update steps in each epoch
    :param num_epochs: number of epoches for which the optimization will run
    :param num_valid_samples: the number of samples in the validation set to test on after each epoch
    """
    i = int(TRAIN_PRCNT * len(images) - 1)
    crop_size = (model.input_shape[1], model.input_shape[2])
    train_set = load_dataset(images[:i], batch_size, corruption_func, crop_size)
    vld_set = load_dataset(images[i:], batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=ADAM_BETA))
    model.fit_generator(generator=train_set, validation_data=vld_set, epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch, validation_steps=num_valid_samples,
                        use_multiprocessing=True)       ## I had to add this because otherwise I got:
                        # RuntimeError: Your generator is NOT thread-safe.Keras requires a thread-safe generator whenuse_multiprocessing=False, workers > 1


def restore_image(corrupted_image, base_model):
    """
    the funcion restores the given corrupted image.
    :param corrupted_image: a grayscale image of shape (height, width)
    :param base_model: a nn trained to restore small patches, its input and output are images with values in range [-0.5,0.5]
    :return: an uncorrupted image
    """
    h, w = corrupted_image.shape
    a = Input(shape=(h, w, 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    corrupted = corrupted_image[np.newaxis, :, :, np.newaxis] - SUB
    predicted = new_model.predict(corrupted)[0].astype(np.float64) + SUB
    predicted = predicted.reshape(h, w)
    return predicted.clip(0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    :param image: a grayscale image with values in range [0,1]
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution
    :param max_sigma: a non-negative scalar ()>=min_sigma) representing the maximal variance of the gaussian distribution
    :return:
    """
    sigma = random.uniform(min_sigma, max_sigma)
    noised = copy.deepcopy(image) + np.random.normal(scale=sigma, size=image.shape)
    noised = np.divide(np.round(noised * MAX_GRAY_SHADE), MAX_GRAY_SHADE)
    return noised.clip(0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    the function should train network
    :param num_res_blocks: number of residual blocks
    :param quick_mode: true for pre-submission
    :return: a trained denoising model
    """
    images = utils.images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        train_model(model, images, noising_corruption_function, 10, 3, 2, 30)
    else:
        train_model(model, images, noising_corruption_function, 100, 100, 5, 1000)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    the function simulate motion blur on the given image using square kernel of size kernel_size where the line has the
    given angle in radians.
    :param image: a grayscale image with values in range [0,1]
    :param kernel_size: an odd integer specify the size of the kernel
    :param angle: an angle in radians in the range [0, pi)
    :return: corrupted image
    """
    kernel = utils.motion_blur_kernel(kernel_size, angle)
    return convolve(copy.deepcopy(image), kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    the function samples an angle at uniform range [0, pi) and chooses a kernel size at uniform from the given list
    :param image: a grayscale image with values in range [0,1]
    :param list_of_kernel_sizes: a list of odd integers
    :return:
    """
    angel = random.uniform(0, np.math.pi)
    ker_idx = int(random.uniform(0, len(list_of_kernel_sizes) - 1))
    corrupted = add_motion_blur(image, list_of_kernel_sizes[ker_idx], angel)
    corrupted = np.divide(np.round(corrupted * MAX_GRAY_SHADE), MAX_GRAY_SHADE)
    return corrupted.clip(0, 1)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    the function will return a trained deblurring model
    :param num_res_blocks: number of residual blocks
    :param quick_mode: true for pre-submission
    :return: a trained deblurring model
    """
    images = utils.images_for_deblurring()
    model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        train_model(model, images, blurring_corruption_function, 10, 3, 2, 30)
    else:
        train_model(model, images, blurring_corruption_function, 100, 100, 10, 1000)
    return model




''' 
# the code to generate graphs
denoise_error = []
deblur_error = []

for i in range(0, 5):
    denoise = learn_denoising_model(i)
    print("denoise ", i, ": ",denoise.input_shape)
    denoise_error.append(denoise.history.history['val_loss'][-1])
    deblur = learn_deblurring_model(i)
    print("deblur ", i, ": ",deblur.input_shape)
    deblur_error.append(deblur.history.history['val_loss'][-1])

arr = np.arange(0, 5)
plt.plot(arr, denoise_error)
plt.xlabel('num res blocks')
plt.ylabel('validation loss denoise')
plt.show()

plt.plot(arr, deblur_error)
plt.xlabel('num res blocks')
plt.ylabel('validation loss deblur')
plt.show()
'''


