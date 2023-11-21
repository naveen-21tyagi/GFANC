import os 
import numpy as np
from scipy import signal
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
from Control_filter_selection import Control_filter_selection
mpl.rcParams['agg.path.chunksize'] = 10000

def loading_paths_from_MAT(folder, Pri_path_file_name, Sec_path_file_name):
    Primay_path_file, Secondary_path_file = os.path.join(folder, Pri_path_file_name), os.path.join(folder, Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path

def resample_wav(waveform, sample_rate, resample_rate):
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform

def loading_real_wave_noise(folde_name, sound_name):
    waveform, sample_rate = torchaudio.load(os.path.join(folde_name, sound_name))
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate) # resample
    return waveform, resample_rate

def Disturbance_generation_from_real_noise(fs, Repet, wave_form, Pri_path, Sec_path):
    wave = wave_form[0,:].numpy()
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec,wave),axis=0) # add the length of the wave_form through repetition
    pass

    # Construting the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec), signal.lfilter(Sec_path, 1, wavec)
    
    N = len(Dir)
    N_z = N//fs
    Dir, Fx = Dir[0:N_z*fs], Fx[0:N_z*fs]
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(wavec).type(torch.float)

class Fixed_filter_controller():
    def __init__(self, Filter_vector, fs):
        self.Filter_vector = torch.from_numpy(Filter_vector).type(torch.float)# torch.Size([Xseconds, 1024])
        Len = self.Filter_vector.shape[1]
        self.fs = fs
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx):
        Error = torch.zeros(Dis.shape[0])
        j = 0
        for ii, dis in enumerate(Dis):
            self.Xd = torch.roll(self.Xd,1,1)
            self.Xd[0,0] = Fx[ii] # Fx[ii]: fixed-x signal
            yt = self.Current_Filter @ self.Xd.t()
            e = dis - yt
            Error[ii] = e.item()
            if (ii + 1) % self.fs == 0:
                self.Current_Filter = self.Filter_vector[j]
                j += 1
        return Error
    
# Loading real noises
mdict = {}
fs = 16000
StepSize = 0.0001
sound_name = 'Aircraft'
waveform, resample_rate = loading_real_wave_noise(folde_name='samples', sound_name=sound_name+'.wav')

# Loading path
Pri_path, Secon_path = loading_paths_from_MAT(folder='paths', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
Dis, Fx, Re = Disturbance_generation_from_real_noise(fs=fs, Repet=0, wave_form=waveform, Pri_path=Pri_path, Sec_path=Secon_path)

# GFANC: present frame to predict the next

MODEL_PTH = 'models/M6_res_Synthetic.pth'
path_mat = 'models/Pretrained_Sub_Control_filters.mat'

# prediction index
Filter_vector = Control_filter_selection(fs=16000, MODEL_PTH=MODEL_PTH, path_mat=path_mat, Primary_noise=Re.unsqueeze(0), threshold=0.5)

Fixed_Cancellation = Fixed_filter_controller(Filter_vector=Filter_vector, fs=16000)
ErrorFixed2 = Fixed_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx)

Time = np.arange(len(Dis))*(1/fs)


plt.title('Aircraft noise')
plt.plot(Time, Dis, color='blue', label='ANC off')
plt.plot(Time, ErrorFixed2, color='orange', label='ANC on')
plt.ylabel('Magnitude')
plt.xlabel('Time (seconds)')
plt.legend()
plt.grid()
plt.show()