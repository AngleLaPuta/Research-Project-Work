import os
import argparse
from scipy.io import loadmat
import scipy.io
from torch.utils.data import DataLoader
from util.reconstructSARimageFFT_3D import reconstructSARimageFFT_3D
from util.dataReadFunction import *
from data_utils import TrainDatasetFromFolder_SAR_adc, ValDatasetFromFolder_SAR_adc, display_transform
import torch
from tqdm import tqdm
import time
from util.helper import convert_array
import cupy as cp

parser = argparse.ArgumentParser(description='Train Super Resolution Models')

parser.add_argument('--config_path', metavar='DIR', default='SAR_config',
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--output', metavar='DIR', default='SAR Image',
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--data_path', metavar='DIR', default='HydraRawData',
                    help='path to output folder. If not set, will be created in data folder') 
parser.add_argument('--high_Azimuth', type=int, default=112, help='hr number of Azimuth')
parser.add_argument('--genetate_step', type=int, default=10, help='Step to generate SAR image')
parser.add_argument('--depth', type=int, default=200, help='Depth of the SAR image')
parser.add_argument('--generate_lr', type=str, default='False') 
parser.add_argument('--low_Azimuth', type=int, default=6*8, help='lr number of Azimuth')


if __name__ == '__main__':
    opt = parser.parse_args()

    adc_filenames = [(filename, os.path.join(root, filename)) for root, dirs, files in os.walk(opt.data_path) for filename in files if filename[-4:] == '.mat']
    adc_filenames = tqdm(adc_filenames)
    for filename, pathfile in adc_filenames:
        # filename = '0_0716_150_1.mat'
        # pathfile = '/localscratch/maolin/projects/Pitt-Radar/dataset/Data/0_0716_150_1.mat'
        
        sensorParams_path = os.path.join(opt.config_path, '1sensorParams.mat')
        sarParams_path = os.path.join(opt.config_path, '1sarParams.mat')
        calData_fileName = os.path.join(opt.config_path, 'calData_2Tx_4Rx.mat')
        delayOffset_fileName = os.path.join(opt.config_path, 'delayOffset_2Tx_4Rx.mat')
        sensorParams = loadmat(sensorParams_path)
        sensorParams = sensorParams['sensorParams']
        sarParams = loadmat(sarParams_path)
        sarParams = sarParams['sarParams']
        calData = loadmat(calData_fileName)
        calData = calData['calData']
        calData = calData.flatten().tolist()
        delayOffset = loadmat(delayOffset_fileName)
        delayOffset = delayOffset['delayOffset']
        delayOffset = delayOffset.flatten().tolist()

        # initialize sensor dictionary
        sensor_dict = {}

        # iterate through the sensorParams dictionary
        for key in sensorParams.dtype.names:
            sensor_dict[key] = sensorParams[0][0][sensorParams.dtype.names.index(key)][0]

        sensorParams = {key: convert_array(val) for key, val in sensor_dict.items()}

        # initialize sar dictionary
        sar_dict = {}

        # iterate through the sarParams dictionary
        for key in sarParams.dtype.names:
            sar_dict[key] = sarParams[0][0][sarParams.dtype.names.index(key)][0]
        
        sarParams = {key: convert_array(val) for key, val in sar_dict.items()}
        rawData, sarParams, sensorParams = dataReadFunction(sensorParams, sarParams, pathfile)

        rawData.reshape(8, 14, 338, 256)

        rawData = calibrateDataFunction(rawData,sensorParams,calData,delayOffset)

        frequency = [
        sensorParams['Start_Freq_GHz'] * 1e9,
        sensorParams['Slope_MHzperus'] * 1e12,
        sensorParams['Sampling_Rate_ksps'] * 1e3,
        sensorParams['Adc_Start_Time_us'] * 1e-6
        ]

        
        c = 299792458
        nFFTkXY = 512
        Samples_per_Chirp = sensorParams['Samples_per_Chirp']
        Num_TX = sensorParams['Num_TX']
        Num_RX = len(sensorParams['RxToEnable'])
        Num_horizontalScan = sarParams['Num_horizontalScan']
        Num_verticalScan = sarParams['Num_verticalScan']

        yStepM_mm = sarParams['Vertical_stepSize_mm']
    
        if Num_horizontalScan != 1:
            if sarParams['Horizontal_stepSize_mm'] == 0:
                # For AMC4030 measurements
                xStepM_mm = sarParams['Platform_Speed_mmps'] * sensorParams['Frame_Repetition_Period_ms'] * 1e-3
            else:
                xStepM_mm = sarParams['Horizontal_stepSize_mm']
        else:
            xStepM_mm = 0

        # Calculate wavelength in mm
        lambda_mm = c / 79e9 * 1e3  # Center frequency
        
        z = filename[:-4].split('_')[2]
        zlist = np.arange(0, opt.depth+1, opt.genetate_step)

        for zTarget in zlist:
            save_folder = os.path.join(opt.output, filename[:-4])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            zTarget_mm = int(z) + zTarget
            save_location = os.path.join(save_folder, f'{filename[:-4]}_{zTarget_mm}')
            rawDataMonostatic = convertMultistaticToMonostatic(rawData, frequency, xStepM_mm, yStepM_mm, zTarget_mm, 'IWR1443', cp.ones(Num_TX), cp.ones(Num_RX))

            rawDataUniform = cp.reshape(rawDataMonostatic, (-1, Num_horizontalScan, Samples_per_Chirp), order='F')    
            reconstructSARimageFFT_3D(save_location, rawDataUniform, frequency, xStepM_mm, lambda_mm/4, zTarget_mm, nFFTkXY, xUpsampleM=1, yUpsampleM=1)