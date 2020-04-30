import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage.interpolation import map_coordinates


GRAY_REP = 1
THREE_DIMS = 3
TOTAL_GRAY_LEVELS = 256
MAX_GRAY_SHADE = 255
CHANGE_RATE_SRT = "change_rate.wav"
CHANGE_SAMPLES_STR = "change_samples.wav"


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


def DFT(signal):
    """
    the function convert 1D discrete signal to its fourier transform
    :param signal: array of type float64 with shape (N,1)
    :return:
    """
    N = signal.shape[0]
    x = np.arange(N)            # x = [0,1,...,N-1]
    u = x.reshape((N, 1))       # current frequency
    e = np.exp(-2j * np.pi * u * x / N)
    return np.dot(e, signal).astype(np.complex128)    # because dot product F = sum(f(x_i) * e_i)


def IDFT(fourier_signal):
    """
    the function convert fourier transform to its 1D discrete signal
    :param fourier_signal: array of type complex128 with shape (N,1)
    :return: complex fourier signal
    """
    N = fourier_signal.shape[0]
    x = np.arange(N)            # x = [0,1,...,N-1]
    u = x.reshape((N, 1))       # current frequency
    e = np.exp(2j * np.pi * u * x / N)
    return (np.dot(e, fourier_signal) / N).astype(np.complex128)

    #if we use just real domain then it would create a mirroring effect
    #because DFT of real values is a mirrored sequence. that's why we don't do np.real()



def DFT2(image):
    """
    the function convert 2D discrete signal to its fourier transform
    :param image: grayscale image of type float64 with shape (N,1)
    :return: complex signal
    """
    im = DFT(image.T)
    return DFT(im.T).astype(np.complex128)


def IDFT2(fourier_image):
    """
    the function convert fourier transform to its 2D discrete signal
    :param fourier_image: 2D array of type complex128 with shape (N,1)
    :return:
    """
    im = IDFT(fourier_image.T)
    return np.real(IDFT(im.T)).astype(np.complex128)


def change_rate(filename, ratio):
    """
    the function changes function rate
    :param filename: path to WAV file
    :param ratio: positive float64 0.25<r<4 representing the duration change
    """
    sr, data = wavfile.read(filename)
    wavfile.write(CHANGE_RATE_SRT, int(sr * ratio), data)


def change_samples(filename, ratio):
    """
    Changes the duration of an audio file by reducing the number of samples
    using fourier.
    :param filename: path to WAV file
    :param ratio: positive float64 number representing sample rate, 0.25<r<4
    :return: 1D ndarray of type float64 representing the new sample points
    """
    sr, data = wavfile.read(filename)
    resized = resize(data, ratio)
    wavfile.write(CHANGE_SAMPLES_STR, sr, resized)
    return resized.astype(np.float64)


def resize(data, ratio):
    """
    the function resize a given set of sample points
    :param data: 1D ndarray of type float64/complex128 representing the original sample points
    :param ratio: positive float64 number representing sample rate, 0.25<r<4
    :return: 1D ndarray of type float64/complex128 representing the new sample points
    """

    if ratio == 1:
        return data

    type = data.dtype
    N1 = data.size
    newdata = np.fft.fftshift(DFT(data))

    if ratio < 1:    # we want to increase number of samples in data
        N2 = int(N1 / ratio)
        half = int((N2 - N1) / 2)
        resized = np.zeros(N2).astype(np.complex128)
        resized[half:-half] = newdata

    else:           # we want to reduce number of pixels in data
        N2 = int(N1 / ratio)
        half = int((N1 - N2) / 2)
        resized = newdata[half:half + N2]

    return IDFT(np.fft.ifftshift(resized)).astype(type)


def resize_spectrogram(data, ratio):
    """
    the function speeds up a wav file.
    :param data: 1D nparray of type float64
    :param ratio: positive float64 number
    :return: a new sample points according to the given ratio.
    """
    mat = stft(data)
    resized = np.array([resize(mat[i], ratio) for i in range(mat.shape[0])])
    y_rec = istft(resized)
    return y_rec


def resize_vocoder(data, ratio):
    """
    the function speeds up a wav file by phase vocoding its spectrogram.
    :param data: 1D ndarray of type float64.
    :param ratio: positive float64 number.
    :return: the given data rescaled according to ratio.
    """
    mat = stft(data)
    voc = phase_vocoder(mat, ratio)
    y_rec = istft(voc)
    return y_rec.astype(np.float64)


def conv_der(im):
    """
    the function computes the magnitude of image derivatives.
    :param im: grayscale float64 image
    :return: the magnitude, with the same type and shape
    """
    kernel = np.array([[0.5, 0, -0.5]])
    dx = signal.convolve2d(im, kernel, mode='same')
    dy = signal.convolve2d(im, kernel.T, mode='same')
    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
    return magnitude.reshape(im.shape).astype(np.float64)


def fourier_der(im):
    """
    the function computes the magnitude of image derivatives using Fourier
    transform.
    :param im: float64 grayscale image
    :return: float64 grayscale magnitude image
    """
    N, M = im.shape
    #dft = DFT2(im)
    dft = np.fft.fft2(im)
    dft = np.fft.fftshift(dft)

    u = np.arange(-N / 2, N / 2).reshape((N, 1))
    v = np.arange(-M / 2, M / 2).reshape((1, M))

    dx = dft * u * 2j * np.pi / N
    dy = dft * v * 2j * np.pi / M

    idx = IDFT2(np.fft.ifftshift(dx))
    idy = IDFT2(np.fft.ifftshift(dy))

    return np.sqrt(np.abs(idx) ** 2 + np.abs(idy) ** 2)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec



