import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from scipy.constants import speed_of_light
import cupy as cp


def dataReadFunction(sensorParams, sarParams, adcBinData_name):
    Chirps_per_Frame = sensorParams['Chirps_per_Frame']
    Num_horizontalScan = sarParams['Num_horizontalScan']
    Num_verticalScan = sarParams['Num_verticalScan']

    if Chirps_per_Frame > 1:
        averageChirps = True
    else:
        averageChirps = False
    
    # Check if the scan mode is Rectangular, Data must be reversed for 2nd, 4th, etc. rows
    if 'scanTrajectoryMode' not in sarParams:
        isRectangularScan = True  # default is Rectangular
    else:
        isRectangularScan = sarParams['scanTrajectoryMode'] == 'Rectangular'
    
    # Check if the Hardware trigger mode is active
    if 'triggerMode' not in sarParams or 'TriggerSelect_Arr' not in sensorParams:
        isHardwareTrigger = False  # Default mode is Software trigger
    else:
        if sarParams['triggerMode'] == 'Hardware' and sensorParams['TriggerSelect_Arr'] == 2:
            isHardwareTrigger = True
        else:
            raise ValueError('SAR and Sensor parameters are inconsistent for the Hardware Trigger mode. Please check the configuration data.')
    
    # Pre-processing of the Hardware Trigger Data
    if isHardwareTrigger:
        if sensorParams['Num_Frames'] == Num_horizontalScan * Num_verticalScan:
            # Total number of frames information is not necessary anymore
            sensorParams['Num_Frames'] = 1
        else:
            raise ValueError('Number of measurements must be equal to the number of frames in the Hardware Trigger continuous mode. Please check the data.')
        
    # Parse Complex rawData (assuming fcn_read_ADC_bin_SAR reads the data and returns a CuPy array)
    rawData = fcn_read_ADC_bin_SAR(sensorParams, sarParams, adcBinData_name)

    if averageChirps:
        # Average Chirps by summing along the 4th axis (axis=3 in Python) and dividing by Chirps_per_Frame
        rawData = cp.sum(rawData, axis=3) / Chirps_per_Frame
    
    Samples_per_Chirp = sensorParams['Samples_per_Chirp']
    Num_TX = sensorParams['Num_TX']
    Num_RX = len(sensorParams['RxToEnable'])
    Num_Frames = sensorParams['Num_Frames']

    # Reshape rawData
    rawData = cp.reshape(rawData, (Num_RX * Num_TX, Samples_per_Chirp, Num_Frames, Num_horizontalScan, Num_verticalScan), order='F')

    # Pre-processing based on SAR Mode
    if sarParams['dataCaptureMode'] == 'Continuous':
        isContinuousMode = True  # If this flag is true, data may need to be cropped.
        
        if not isHardwareTrigger:
            # Pre-process data for the Software Trigger mode (Data is already ready for the Hardware Trigger mode)
            if Num_horizontalScan != 1:
                raise ValueError('Number of horizontal scan must be 1 in Software Trigger Continuous mode. Please check the data')

            Num_horizontalScan = Num_Frames
            Num_Frames = 1
            sarParams['Num_horizontalScan'] = Num_horizontalScan
            sensorParams['Num_Frames'] = Num_Frames

    elif sarParams['dataCaptureMode'] == 'Discrete':
        isContinuousMode = False  # Data will not be cropped in these modes

        if Num_Frames != 1:
            answerAverageFrames = input("Data cannot be used for image reconstruction directly. Do you want to average over frames? (Yes/No): ")

            if answerAverageFrames.lower() == 'yes':
                # Average over frames
                rawData = cp.sum(rawData, axis=2) / Num_Frames
                Num_Frames = 1
                sensorParams['Num_Frames'] = Num_Frames
            elif answerAverageFrames.lower() == 'no':
                print('Data cannot be used for image reconstruction directly, please reshape it manually.')
                # Convert rawData to: (Num_RX * Num_TX) * Num_verticalScan * Num_horizontalScan * Num_Frames * Samples_per_Chirp
                rawData = cp.transpose(rawData, (0, 4, 3, 2, 1))
                # Returning rawData after permuting
                return rawData
            else:
                raise ValueError("Invalid response. Please input 'Yes' or 'No'.")

    elif sarParams['dataCaptureMode'] == 'Stationary':
        print('Data cannot be used for image reconstruction directly, please reshape it manually.')
        # Convert rawData to: (Num_RX * Num_TX) * Num_verticalScan * Num_horizontalScan * Num_Frames * Samples_per_Chirp
        rawData = cp.transpose(rawData, (0, 4, 3, 2, 1))
        # Returning rawData after permuting
        return rawData

    else:
        raise ValueError('Data capture mode is wrong. Please check the data.')
    
    # Reshape rawData
    rawData = cp.reshape(rawData, (Num_RX * Num_TX, Samples_per_Chirp, Num_horizontalScan, Num_verticalScan), order='F')

    # For Continuous Scan - Crop the Data
    if isContinuousMode and not isHardwareTrigger:
        
        # Check the Trigger Offset Time
        if 'Trigger_timeOffset_s' in sarParams:
            if sarParams['Trigger_timeOffset_s'] < 0:  # If Trigger is ahead of the scan
                firstIndex = int(cp.floor(abs(sarParams['Trigger_timeOffset_s']) / (sensorParams['Frame_Repetition_Period_ms'] * 1e-3)) + 1)
                motionTime_s = calculateMotionDuration(sarParams['Horizontal_scanSize_mm'], sarParams['Platform_Speed_mmps'], 200)
                lastIndex = int(cp.ceil(motionTime_s / (sensorParams['Frame_Repetition_Period_ms'] * 1e-3)) + firstIndex)
            else:
                firstIndex = 1
                lastIndex = Num_horizontalScan  # Get all the samples
        else:
            # Manually crop the data by data analysis
            firstIndex = 16
            lastIndex = 423

        # Crop the Data
        rawData = rawData[:, :, firstIndex-1:lastIndex, :]

        # Define new Num_horizontalScan, rearrange the data
        _, _, Num_horizontalScan, _ = rawData.shape
        sarParams['Num_horizontalScan'] = Num_horizontalScan
    
    if isRectangularScan:
        for n in range(Num_verticalScan):
            if (n % 2) == 1:  # Check if n is odd
                rawData[:, :, :, n] = cp.flip(rawData[:, :, :, n], axis=2)

    # Reshape the rawData
    # Convert rawData to: (Num_RX * Num_TX) * Num_verticalScan * Num_horizontalScan * Samples_per_Chirp
    rawData = cp.transpose(rawData, (0, 3, 2, 1))
    return rawData, sarParams, sensorParams


def fcn_read_ADC_bin_SAR(sensorParams, sarParams, adcBinData_name):
    Num_RX_channels = 4
    Samples_per_Chirp = sensorParams['Samples_per_Chirp']
    Chirps_per_Frame = sensorParams['Chirps_per_Frame']
    Num_TX = sensorParams['Num_TX']
    Num_Frames = sensorParams['Num_Frames']
    Num_measurements = sarParams['Num_horizontalScan'] * sarParams['Num_verticalScan']

    # Parse the binary data
    ErrStatus, rawData = Parse_Datafile_bin_SAR(adcBinData_name, 2*Num_RX_channels, Samples_per_Chirp, Chirps_per_Frame*Num_TX, Num_Frames, Num_measurements)
    
    # Convert rawData to CuPy array for GPU processing
    rawData = cp.asarray(rawData.flatten())

    if ErrStatus != 0:
        raise ValueError('Error in reading the ADC data file. Please check the data file.')
    
    # Reshape rawData to 2*Num_RX_channels rows and appropriate number of columns
    rawData = cp.reshape(rawData, (2 * Num_RX_channels, -1), order='F')

    # Combine real and imaginary parts of rawData
    rawData_Rxchain = rawData[[0, 2, 4, 6], :] + 1j * rawData[[1, 3, 5, 7], :]

    # Reshape rawData_Rxchain
    rawData_Rxchain = cp.reshape(rawData_Rxchain, (Num_RX_channels, Samples_per_Chirp, Num_TX, Chirps_per_Frame, Num_Frames, Num_measurements), order='F')

    # Permute the dimensions to match the desired order: Num_RX_channels x Num_TX x Samples_per_Chirp x Chirps_per_Frame x Num_Frames x Num_measurements
    rawData_Rxchain = cp.transpose(rawData_Rxchain, (0, 2, 1, 3, 4, 5))

    return rawData_Rxchain


def Parse_Datafile_bin_SAR(adc_file_name, Num_channels, Samples_per_Chirp, Chirps_per_Frame, Num_frames, Num_measurements):
    Expected_Num_Samples = Num_channels * Samples_per_Chirp * Chirps_per_Frame * Num_frames * Num_measurements
    
    # Load the .mat file using scipy (this part is done on CPU because the file is loaded from disk)
    radar_data = scipy.io.loadmat(adc_file_name)
    radar_data = radar_data['adc_data_total']
    
    # Check if the number of samples matches the expected number
    if radar_data.size != Expected_Num_Samples:
        print('Number of samples in data file not matching expected')
        ErrStatus = -2
    else:
        ErrStatus = 0  # No Error
    
    # Return the data as a numpy array since CuPy cannot directly read .mat files
    return ErrStatus, radar_data
    


def calculateMotionDuration(distance_mm,speed_mmps,acceleration_mmps2):
    # Calculate ramp-up time and distance
    rampupTime_s = speed_mmps / acceleration_mmps2
    rampupDistance_mm = 0.5 * acceleration_mmps2 * rampupTime_s**2

    # Ensure distance is positive
    distance_mm = abs(distance_mm)

    # If the distance is enough to reach the maximum speed
    if distance_mm >= 2 * rampupDistance_mm:
        # Calculate the constant speed running time
        runningDistance_mm = distance_mm - 2 * rampupDistance_mm
        runningTime_s = runningDistance_mm / speed_mmps
        motionTime_s = runningTime_s + 2 * rampupTime_s
        return motionTime_s
    else:
        # Recalculate ramp-up time if the distance is shorter
        rampupDistance_mm = distance_mm / 2
        rampupTime_s = np.sqrt(2 * rampupDistance_mm / acceleration_mmps2)
        motionTime_s = 2 * rampupTime_s
        return motionTime_s
    

def calibrateDataFunction(rawData, sensorParams, calData, delayOffset):
    """
    Calibrate the raw data based on complex gain and beat frequency offset.

    Parameters:
    rawData: ndarray (CuPy array)
        The raw data array with shape (Num_RX * Num_TX, Num_verticalScan, Num_horizontalScan, Samples_per_Chirp)
    sensorParams: dict
        The sensor parameters including Slope_MHzperus, Sampling_Rate_ksps, and Samples_per_Chirp.
    calData: ndarray (CuPy array)
        Calibration data array with shape (Num_RX * Num_TX, 1)
    delayOffset: ndarray (CuPy array)
        Delay offset array with shape (Num_RX * Num_TX, 1)

    Returns:
    rawDataCal: ndarray (CuPy array)
        Calibrated data array.
    """

    # Complex Gain Calibration
    calData = cp.asarray(calData)
    calData = calData[:, cp.newaxis, cp.newaxis, cp.newaxis]
    rawDataCal = calData * rawData

    # Beat Frequency Offset Calibration
    Slope_Hzpers = sensorParams['Slope_MHzperus'] * 1e12
    Sampling_Rate_sps = sensorParams['Sampling_Rate_ksps'] * 1e3
    Samples_per_Chirp = sensorParams['Samples_per_Chirp']

    # Create wideband frequency array
    f = (cp.arange(Samples_per_Chirp) * Slope_Hzpers / Sampling_Rate_sps)
    f = cp.reshape(f, (1, 1, 1, -1))

    # Compute the frequency bias factor
    delayOffset = cp.asarray(delayOffset)
    frequencyBiasFactor = cp.exp(-1j * 2 * cp.pi * delayOffset[:, None, None, None] * f)

    # Apply frequency bias correction
    rawDataCal = rawDataCal * frequencyBiasFactor

    return rawDataCal



def convertMultistaticToMonostatic(sarDataMultistatic, chirpParameters, xStepM, yStepM, zTarget, radarType, activeTx, activeRx):
    """
    Converts multistatic SAR data to monostatic SAR data using GPU-accelerated CuPy.
    
    Parameters:
    sarDataMultistatic: ndarray (CuPy array)
        Multistatic SAR data of shape (nChannel, yPointM, xPointM, nSample)
    chirpParameters: list
        [fStart, fSlope, fSample, adcStart]
    xStepM: float
        Measurement step size in the x (horizontal) axis in mm
    yStepM: float
        Measurement step size in the y (vertical) axis in mm
    zTarget: float
        Target distance in mm
    radarType: str
        Type of radar ('IWR1443', '4ChipCascade', 'Simulation')
    activeTx: list
        Active Tx antennas
    activeRx: list
        Active Rx antennas

    Returns:
    sarDataMonostatic_Ver1: ndarray (CuPy array)
        Monostatic SAR data with phase correction (version 1)
    sarDataMonostatic_Ver2: ndarray (CuPy array)
        Monostatic SAR data with phase correction (version 2)
    """
    
    # Define Frequency Spectrum
    nChannel, yPointM, xPointM, nSample = sarDataMultistatic.shape
    # new_xPointM = 512
    # new_yPointM = 14

    # sarDataMultistatic_padded = cp.zeros((sarDataMultistatic.shape[0], new_yPointM, new_xPointM, sarDataMultistatic.shape[3]), dtype=cp.complex64)
    
    # x_start = (new_xPointM - xPointM) // 2
    # y_start = (new_yPointM - yPointM) // 2
    # sarDataMultistatic_padded[:, y_start:y_start+yPointM, x_start:x_start+xPointM, :] = sarDataMultistatic

    # sarDataMultistatic = sarDataMultistatic_padded
    # xPointM = new_xPointM 
    # yPointM = new_yPointM

    if len(chirpParameters) > 1 and len(chirpParameters) <= 4 and nSample > 1:
        f0, K, fS, adcStart = chirpParameters
        f0 = f0 + adcStart * K  # Apply ADC sampling offset
        f = f0 + cp.arange(nSample) * K / fS  # Wideband frequency
    elif len(chirpParameters) == 1 and nSample == 1:
        f = chirpParameters[0]
    else:
        raise ValueError("Please correct the frequency configuration and data")

    # Define Fixed Parameters
    c = 299792458  # Speed of light in m/s
    k = 2 * cp.pi * f / c
    k = cp.reshape(k, (1, 1, 1, -1))

    # Check data size and active channels
    nTx = cp.sum(activeTx)
    nRx = cp.sum(activeRx)
    if nChannel != (nTx * nRx):
        raise ValueError("Please correct the active channel data")

    # Antenna Locations and distances
    rxAntPos, txAntPos, virtualChPos, d_r = getAntennaLocations(radarType)  # Keep this on the CPU side
    rxAntPos, txAntPos, virtualChPos = cropAntennaLocationsSetArrayCenter(rxAntPos, txAntPos, virtualChPos, activeTx, activeRx)

    # Define Measurement Locations at Linear Rail
    xAxisM = xStepM * (-(xPointM - 1) / 2 + cp.arange(xPointM)) * 1e-3  # Convert step size from mm to meters
    yAxisM = yStepM * (-(yPointM - 1) / 2 + cp.arange(yPointM)) * 1e-3
    zAxisM = cp.array([0])

    zM, xM, yM = cp.meshgrid(zAxisM, xAxisM, yAxisM)
    xyzM = cp.concatenate([xM, yM, zM], axis=1)
    xyzM = cp.reshape(cp.transpose(xyzM, (0, 2, 1)), (-1, 3), order='F')
    nMeasurement = xyzM.shape[0]

    # Define Target Locations
    xyzT = cp.array([-400, 0, zTarget]) * 1e-3  # Target in meters

    # Multistatic to Monostatic Phase Correction
    nTx = int(nTx)
    nRx = int(nRx)
    txAntPos = np.tile(txAntPos, (nRx, 1)).reshape(nTx, -1, 3, order='F').transpose(1, 0, 2).reshape(-1, 3, order='F')
    rxAntPos = np.tile(rxAntPos, (nTx, 1))

    txAntPos = txAntPos.reshape(nChannel, 1, 3, order='F')
    rxAntPos = rxAntPos.reshape(nChannel, 1, 3, order='F')

    xyzM = xyzM.reshape(1, nMeasurement, 3, order='F')
    xyzM = cp.tile(xyzM, (nChannel, 1, 1))

    xyzM_Tx = xyzM + cp.asarray(txAntPos)
    xyzM_Rx = xyzM + cp.asarray(rxAntPos)
    xyzM_Tx = xyzM_Tx.reshape(-1, 3, order='F')
    xyzM_Rx = xyzM_Rx.reshape(-1, 3, order='F')

    virtualChPos = cp.asarray(virtualChPos).reshape(nChannel, 1, 3, order='F')
    xyzM_TRx = xyzM + virtualChPos
    xyzM_TRx = xyzM_TRx.reshape(-1, 3, order='F')

    # Distance matrix for multistatic (using CPU because cdist is not supported by CuPy)
    if xyzT.ndim == 1:
        xyzT = cp.asnumpy(xyzT).reshape(1, 3, order='F')  # Ensure it's a 2D array with shape (1, 3)
    R_Tx_T = cdist(cp.asnumpy(xyzM_Tx), xyzT)
    R_Rx_T = cdist(cp.asnumpy(xyzM_Rx), xyzT)

    # Distance matrix for monostatic
    R_TRx_T = 2 * cdist(cp.asnumpy(xyzM_TRx), xyzT)

    # Signal reference multistatic
    k = cp.squeeze(k).T
    signalRefMultistatic = cp.exp(1j * (cp.asarray(R_Tx_T) + cp.asarray(R_Rx_T)) * k)
    signalRefMultistatic = signalRefMultistatic.reshape(nChannel, xPointM, yPointM, nSample, order='F')
    signalRefMultistatic = cp.transpose(signalRefMultistatic, (0, 2, 1, 3))

    # Signal reference monostatic
    signalRefMonostatic = cp.exp(1j * cp.asarray(R_TRx_T) * k)
    signalRefMonostatic = signalRefMonostatic.reshape(nChannel, xPointM, yPointM, nSample, order='F')
    signalRefMonostatic = cp.transpose(signalRefMonostatic, (0, 2, 1, 3))

    sarDataMonostatic_Ver2 = sarDataMultistatic * signalRefMonostatic / signalRefMultistatic

    return sarDataMonostatic_Ver2

import cupy as cp
from scipy.spatial.distance import cdist

def recoverSARData(sarDataMultistatic, chirpParameters, xStepM, yStepM, zTarget, radarType, activeTx, activeRx):
 
    # Define Frequency Spectrum
    nChannel, yPointM, xPointM, nSample = sarDataMultistatic.shape
    # Prepare Recovery Grid
    nRows = 14
    recoveredSARData = cp.zeros((nChannel, nRows, xPointM, nSample), dtype=cp.complex64)

    if len(chirpParameters) > 1 and len(chirpParameters) <= 4 and nSample > 1:
        f0, K, fS, adcStart = chirpParameters
        f0 = f0 + adcStart * K  # Apply ADC sampling offset
        f = f0 + cp.arange(nSample) * K / fS  # Wideband frequency
    elif len(chirpParameters) == 1 and nSample == 1:
        f = chirpParameters[0]
    else:
        raise ValueError("Please correct the frequency configuration and data")

    # Define Fixed Parameters
    c = 299792458  # Speed of light in m/s
    k = 2 * cp.pi * f / c
    k = cp.reshape(k, (1, 1, 1, -1))

    # Check data size and active channels
    nTx = cp.sum(activeTx)
    nRx = cp.sum(activeRx)
    if nChannel != (nTx * nRx):
        raise ValueError("Please correct the active channel data")

    # Antenna Locations and distances
    rxAntPos, txAntPos, virtualChPos, d_r = getAntennaLocations(radarType)  # Keep this on the CPU side
    rxAntPos, txAntPos, virtualChPos = cropAntennaLocationsSetArrayCenter(rxAntPos, txAntPos, virtualChPos, activeTx, activeRx)

    
    # Multistatic to Monostatic Phase Correction
    nTx = int(nTx)
    nRx = int(nRx)
    txAntPos = np.tile(txAntPos, (nRx, 1)).reshape(nTx, -1, 3, order='F').transpose(1, 0, 2).reshape(-1, 3, order='F')
    rxAntPos = np.tile(rxAntPos, (nTx, 1))

    txAntPos = txAntPos.reshape(nChannel, 1, 3, order='F')
    rxAntPos = rxAntPos.reshape(nChannel, 1, 3, order='F')

    
    # Define Measurement Locations at Linear Rail
    xAxisM = xStepM * (-(xPointM - 1) / 2 + cp.arange(xPointM)) * 1e-3  # Convert step size from mm to meters
    yAxisM_all = yStepM * (-(nRows - 1) / 2 + cp.arange(nRows)) * 1e-3
    for i in range(nRows):
        yAxisM = cp.array([yAxisM_all[i]])
        zAxisM = cp.array([0])
        zM, xM, yM = cp.meshgrid(zAxisM, xAxisM, yAxisM)
        xyzM = cp.concatenate([xM, yM, zM], axis=1)
        xyzM = cp.reshape(cp.transpose(xyzM, (0, 2, 1)), (-1, 3), order='F')
        nMeasurement = xyzM.shape[0]


        xyzM = xyzM.reshape(1, nMeasurement, 3, order='F')
        xyzM = cp.tile(xyzM, (nChannel, 1, 1))

        xyzM_Tx = xyzM + cp.asarray(txAntPos)
        xyzM_Rx = xyzM + cp.asarray(rxAntPos)
        xyzM_Tx = xyzM_Tx.reshape(-1, 3, order='F')
        xyzM_Rx = xyzM_Rx.reshape(-1, 3, order='F')

        virtualChPos = cp.asarray(virtualChPos).reshape(nChannel, 1, 3, order='F')
        xyzM_TRx = xyzM + virtualChPos
        xyzM_TRx = xyzM_TRx.reshape(-1, 3, order='F')

        y_val = cp.asarray(yAxisM_all[i])  # Ensure CuPy array
        z_val = cp.asarray(zTarget)  # Ensure CuPy array
        zero_val = cp.asarray(0, dtype=cp.float32)  # Convert scalar to CuPy array
        xyzT = cp.array([y_val, zero_val, z_val]) * 1e-3
        if xyzT.ndim == 1:
            xyzT = cp.asnumpy(xyzT).reshape(1, 3, order='F')  # Ensure it's a 2D array with shape (1, 3)
        R_Tx_T = cdist(cp.asnumpy(xyzM_Tx), xyzT)
        R_Rx_T = cdist(cp.asnumpy(xyzM_Rx), xyzT)

        # Distance matrix for monostatic
        R_TRx_T = 2 * cdist(cp.asnumpy(xyzM_TRx), xyzT)

        # Signal reference multistatic
        k = cp.squeeze(k).T
        signalRefMultistatic = cp.exp(1j * (cp.asarray(R_Tx_T) + cp.asarray(R_Rx_T)) * k)
        signalRefMultistatic = signalRefMultistatic.reshape(nChannel, xPointM, yPointM, nSample, order='F')
        signalRefMultistatic = cp.transpose(signalRefMultistatic, (0, 2, 1, 3))

        # Signal reference monostatic
        signalRefMonostatic = cp.exp(1j * cp.asarray(R_TRx_T) * k)
        signalRefMonostatic = signalRefMonostatic.reshape(nChannel, xPointM, yPointM, nSample, order='F')
        signalRefMonostatic = cp.transpose(signalRefMonostatic, (0, 2, 1, 3))

        phaseCorrectedData = sarDataMultistatic * signalRefMonostatic / (signalRefMultistatic + 1e-6)
        phaseCorrectedData = phaseCorrectedData / cp.linalg.norm(phaseCorrectedData, axis=-1, keepdims=True)
    
        recoveredSARData[:, i:i+1, :, :] = phaseCorrectedData
        # recoveredSARData[:,i:i+1,:,:] = sarDataMultistatic * signalRefMonostatic / signalRefMultistatic

    return recoveredSARData






def getAntennaLocations(radarType='IWR1443', isFigure=False):
    """
    Get the antenna locations based on the radar type using CuPy for GPU acceleration.
    
    Parameters:
    radarType: str
        Type of radar ('IWR1443', '2ChipCascade', 'Simulation')
    isFigure: bool
        Whether to plot the antenna locations or not (not used here).
        
    Returns:
    rxAntPos: ndarray (CuPy array)
        Receive antenna positions
    txAntPos: ndarray (CuPy array)
        Transmit antenna positions
    virtualChPos: ndarray (CuPy array)
        Virtual channel positions
    distAntennas: ndarray (CuPy array)
        Distances between antennas
    """
    
    # Speed of light
    c = 299792458  # Speed of light in m/s

    if radarType == 'IWR1443':
        fC = 79e9  # Center frequency in Hz
        lambda_c = c / fC
        
        dTxRx = 9e-3  # Distance between TX and RX antennas in meters
        
        # Rx Antenna Positions (in meters)
        rxAntPos = cp.array([[0, 0, 0],
                             [0, lambda_c / 2, 0],
                             [0, lambda_c, 0],
                             [0, 3 * lambda_c / 2, 0]])
        
        # Tx Antenna Positions (in meters)
        txAntPos = cp.array([[0, 3 * lambda_c / 2 + dTxRx, 0],
                             [0, 3 * lambda_c / 2 + dTxRx + 2 * lambda_c, 0]])
        
    elif radarType == '2ChipCascade':
        fC = 77e9  # Center frequency in Hz
        lambda_c = c / fC
        
        # Rx Antenna Positions (in meters)
        yAxisRx = cp.arange(0, 28) * lambda_c / 2
        rxAntPos = cp.column_stack([cp.zeros(8), yAxisRx[[0, 1, 2, 3, 24, 25, 26, 27]], cp.zeros(8)])
        
        # Tx Antenna Positions (in meters)
        yAxisTx = cp.arange(0, 6) * 2 * lambda_c
        txAntPos = cp.column_stack([lambda_c / 2 * cp.ones(6), 7 * lambda_c / 4 + yAxisTx, cp.zeros(6)])
    
    elif radarType == 'Simulation':
        fC = 77e9  # Center frequency in Hz
        lambda_c = c / fC
        
        # Rx Antenna Positions (in meters)
        yAxisRxBlock = cp.arange(0, 8) * lambda_c / 2
        rxAntPos = cp.column_stack([cp.zeros(16), cp.concatenate([yAxisRxBlock, yAxisRxBlock + 48 * lambda_c]), cp.zeros(16)])
        
        # Tx Antenna Positions (in meters)
        yAxisTx = cp.arange(0, 12) * 4 * lambda_c
        txAntPos = cp.column_stack([cp.zeros(12), yAxisTx, cp.zeros(12)])
    
    else:
        raise ValueError('Please enter a correct radar type.')
    
    # Calculate virtual channel positions and antenna distances
    nRx = rxAntPos.shape[0]
    nTx = txAntPos.shape[0]
    
    txT = cp.reshape(txAntPos, (nTx, 1, 3), order='F')
    rxT = cp.reshape(rxAntPos, (1, nRx, 3), order='F')
    
    virtualChPos = (txT + rxT) / 2  # Midpoint between Tx and Rx antennas
    virtualChPos = cp.reshape(cp.transpose(virtualChPos, (1, 0, 2)), (-1, 3), order='F')
    
    distAntennas = txT - rxT  # Distance between Tx and Rx antennas
    distAntennas = cp.reshape(cp.transpose(distAntennas, (1, 0, 2)), (-1, 3), order='F')
    
    return rxAntPos, txAntPos, virtualChPos, distAntennas


def cropAntennaLocationsSetArrayCenter(rxAntPos, txAntPos, virtualChPos, activeTx, activeRx):
    """
    Crop antenna locations and set the array center to (0,0,0) using CuPy for GPU acceleration.
    
    Parameters:
    rxAntPos: ndarray (CuPy array)
        Receive antenna positions.
    txAntPos: ndarray (CuPy array)
        Transmit antenna positions.
    virtualChPos: ndarray (CuPy array)
        Virtual channel positions.
    activeTx: ndarray (CuPy array)
        Active transmit antennas (boolean array).
    activeRx: ndarray (CuPy array)
        Active receive antennas (boolean array).
        
    Returns:
    rxAntPos: ndarray (CuPy array)
        Cropped and centered receive antenna positions.
    txAntPos: ndarray (CuPy array)
        Cropped and centered transmit antenna positions.
    virtualChPos: ndarray (CuPy array)
        Cropped and centered virtual channel positions.
    """

    # Crop virtual antenna positions
    virtualChPos = cp.reshape(virtualChPos, (len(activeRx), len(activeTx), -1), order='F')
    virtualChPos = virtualChPos[activeRx == 1][:, activeTx == 1, :]
    virtualChPos = cp.reshape(virtualChPos, (-1, 3), order='F')

    # Crop transmit and receive antenna positions
    txAntPos = txAntPos[cp.where(activeTx)[0], :]
    rxAntPos = rxAntPos[cp.where(activeRx)[0], :]

    # Set the array center (Array virtual center will be at (0,0,0))
    xOffsetArray = (cp.max(virtualChPos[:, 0]) + cp.min(virtualChPos[:, 0])).item() / 2
    yOffsetArray = (cp.max(virtualChPos[:, 1]) + cp.min(virtualChPos[:, 1])).item() / 2
    xyzOffsetArray = cp.array([xOffsetArray, yOffsetArray, 0])

    # Adjust positions by subtracting the offset
    txAntPos -= xyzOffsetArray
    rxAntPos -= xyzOffsetArray
    virtualChPos -= xyzOffsetArray

    return rxAntPos, txAntPos, virtualChPos