import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage import exposure
import cupy as cp  # Import CuPy
from matplotlib import pyplot as plt

def reconstructSARimageFFT_3D(save_location, collectData, frequency, xStepM, yStepM, zTarget, nFFTkXY, xUpsampleM=1, yUpsampleM=1):
    isAmplitudeFactor = True
    is2DImaging = True

    sarData = collectData

    # Define Frequency Spectrum
    nSample = 256
    if (len(frequency) > 1) and (len(frequency) <= 4) and (nSample > 1):
        f0, K, fS, adcStart = frequency
        f0 = f0 + adcStart * K  # ADC sampling offset
        f = f0 + cp.arange(nSample) * K / fS  # Wideband frequency
        # f = f[freq_range]
    elif (len(frequency) == 1) and (nSample == 1) and (is2DImaging == True):
        f = cp.asarray(frequency)
    else:
        raise ValueError('Please correct the configuration and data for 3D processing')

    c = 299792458
    k = 2 * cp.pi * f / c
    k = k.reshape((1, 1, -1))

    yPointM, xPointM, _ = sarData.shape
    xStepT = xStepM / xUpsampleM
    yStepT = yStepM / yUpsampleM
    zRangeT_mm = zTarget * 1e-3
    
    # Define Number of FFT Points
    if (nFFTkXY < xPointM) or (nFFTkXY < yPointM):
        print("Warning: # of FFT points should be greater than the # of measurement points. FFT will be performed at # of measurement points")

    # Set nFFTkX and nFFTkY accordingly
    nFFTkX = max(nFFTkXY, xPointM)
    nFFTkY = max(nFFTkXY, yPointM)
    # nFFTkX = xPointM
    # nFFTkY = yPointM

    xRangeT_mm = xStepT * np.arange(-(nFFTkX - 1) / 2, (nFFTkX - 1) / 2 + 1)
    yRangeT_mm = yStepT * np.arange(-(nFFTkY - 1) / 2, (nFFTkY - 1) / 2 + 1)
    # Define Wavenumbers
    wSx = 2 * cp.pi / (xStepT * 1e-3)  # Sampling frequency for Target Domain
    kX = cp.linspace(-(wSx / 2), (wSx / 2), nFFTkX).reshape(1, -1, 1)  # Reshaped to 1 x nFFTkX x 1

    wSy = 2 * cp.pi / (yStepT * 1e-3)
    kY = cp.linspace(-(wSy / 2), (wSy / 2), nFFTkY).reshape(-1, 1, 1)  # Reshaped to nFFTkY x 1 x 1

    # Zero Padding to sarData to Locate Target at Center
    sarDataPadded = sarData.astype(cp.complex64)


    # Padding for x dimension
    pad_x = nFFTkX - xPointM
    pad_x_pre = pad_x // 2
    pad_x_post = pad_x - pad_x_pre
    sarDataPadded = cp.pad(sarDataPadded, ((0, 0), (pad_x_pre, pad_x_post), (0, 0)), 'constant', constant_values=0)

    # Padding for y dimension
    pad_y = nFFTkY - yPointM
    pad_y_pre = pad_y // 2
    pad_y_post = pad_y - pad_y_pre
    sarDataPadded = cp.pad(sarDataPadded, ((pad_y_pre, pad_y_post), (0, 0), (0, 0)), 'constant', constant_values=0)

    # Calculate kZ
    k_squared = (2 * k) ** 2
    kZ_squared = k_squared - kX ** 2 - kY ** 2
    kZ = cp.sqrt(kZ_squared.astype(cp.complex64))


    # Take 2D FFT of SAR Data
    sarDataFFT = cp.fft.fftshift(cp.fft.fft2(sarDataPadded, axes=(0, 1)), axes=(0, 1))

    # Create 2D-SAR Image for single Z
    if is2DImaging:
        phaseFactor = cp.exp(-1j * zRangeT_mm * kZ)
        phaseFactor[(kX ** 2 + kY ** 2) > k_squared] = 0

        if isAmplitudeFactor:
            sarDataFFT *= kZ

        sarDataFFT *= phaseFactor
        sarDataFFT = cp.sum(sarDataFFT, axis=2)
        sarImage = cp.fft.ifft2(sarDataFFT)


    sarImage = cp.flip(sarImage, axis=1)
    mean_v = np.mean(cp.abs(cp.squeeze(sarImage)))

    sarImage_abs = cp.abs(cp.squeeze(sarImage))
    sarImage_abs_cpu = cp.asnumpy(sarImage_abs)  

    sarImage_norm = exposure.rescale_intensity(sarImage_abs_cpu, out_range=(0, 1))
    save_file = img_as_ubyte(sarImage_norm)

    plt.figure(figsize=(8, 8))
    plt.imshow(save_file, cmap='gray', origin='lower')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_location, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
