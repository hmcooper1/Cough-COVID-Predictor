# This script includes adapted code from:
# https://github.com/skanderhamdi/attention_cnn_lstm_covid_mel_spectrogram
# Original license and authorship belong to the original contributors

# Import libraries =============================================================
import pandas as pd
import numpy as np
import librosa
import os
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from PIL import Image


# Function for showing a progress bar ==========================================
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    printProgressBar(0)
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()


# Function for augmenting audio and creating melspectograms ====================
def SpectAugment(waves_path, files, param_masking, mels_path, labels_path, mean_signal_length):
  
    labels_list = []
    count = 0
    meanSignalLength = mean_signal_length
    
    for fn in progressBar(files, prefix = 'Converting:', suffix = '', length = 50):
        if fn == '.DS_Store':
            continue
        label = fn.split('.')[0].split('_')[1]
        signal , sr = librosa.load(waves_path+fn)
        s_len = len(signal)
        
        # Add zero padding to the signal if less than 156027 (~4.07 seconds)
        # Remove from begining and the end if signal length is greater than 156027 (~4.07 seconds)
        if s_len < meanSignalLength:
               pad_len = meanSignalLength - s_len
               pad_rem = pad_len % 2
               pad_len //= 2
               signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
        else:
               pad_len = s_len - meanSignalLength
               pad_len //= 2
               signal = signal[pad_len:pad_len + meanSignalLength]
        label = fn.split('.')[0].split('_')[1]
        mel_spectrogram = librosa.feature.melspectrogram(y=signal,sr=sr,n_mels=128,hop_length=512,fmax=8000,n_fft=512,center=True)
        dbscale_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max,top_db=80)
        
        plt.figure(figsize=(5.15, 1.99), dpi=100)  # Dimensions: 515 x 199
        plt.imshow(dbscale_mel_spectrogram, interpolation='nearest', origin='lower', aspect='auto')
        plt.axis('off')
        plt.savefig(mels_path + str(count) + ".png", dpi=100)
        plt.close()
        img = Image.open(mels_path + str(count) + ".png").convert("RGB")
        img.save(mels_path + str(count) + ".png")
        
        # Save image names with corresponding labels (0: no COVID, 1: COVID)
        labels_list.append({'filename': f"{count}.png", 'label': label})
        count+=1
        
        if label == '1': # if COVID
            freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=param_masking)
            time_mask = tfio.audio.time_mask(freq_mask, param=param_masking)

            plt.figure(figsize=(5.15, 1.99), dpi=100)  # 515 x 199
            plt.imshow(time_mask, interpolation='nearest', origin='lower', aspect='auto')
            plt.axis('off')
            plt.savefig(mels_path + str(count) + ".png", dpi=100)
            plt.close()
            img = Image.open(mels_path + str(count) + ".png").convert("RGB")
            img.save(mels_path + str(count) + ".png")
            
            # Save image names with corresponding labels (0: no COVID, 1: COVID)
            labels_list.append({'filename': f"{count}.png", 'label': label})
            count+=1

        freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=param_masking)
        time_mask = tfio.audio.time_mask(freq_mask, param=param_masking)
        
        plt.figure(figsize=(5.15, 1.99), dpi=100) # 515 x 199
        plt.imshow(time_mask, interpolation='nearest', origin='lower', aspect='auto')
        plt.axis('off')
        plt.savefig(mels_path + str(count) + ".png", dpi=100)
        plt.close()
        img = Image.open(mels_path + str(count) + ".png").convert("RGB")
        img.save(mels_path + str(count) + ".png")
        
        # Save image names with corresponding labels (0: no COVID, 1: COVID)
        labels_list.append({'filename': f"{count}.png", 'label': label})
        count+=1
    
    # Save labels
    Y = pd.DataFrame(labels_list)
    Y.to_csv(labels_path,index=False)


# Define filepaths and call function ===========================================
# Manually define mean_signal_length
mean_signal_length = 181201

# Filepath for augmented audio (original and pitch shifted audio): input
wavs_signal_augmented = "/rds/general/project/hda_24-25/live/ML/Group14/data_augmentation/final_audio/"
files = os.listdir(wavs_signal_augmented)
files = [f for f in files if f.endswith('.wav')]

# Filepath for where to save melspectograms: output
augmentedData = "/rds/general/project/hda_24-25/live/ML/Group14/data_augmentation/melspectograms/"

# Filepath for where labels (0: no COVID, 1: COVID) for each melspectogram will be saved
labels_mels_signal_augmented = "/rds/general/project/hda_24-25/live/ML/Group14/data_augmentation/labels.csv"

# Call function
SpectAugment(wavs_signal_augmented,
             files,
             30,
             augmentedData,
             labels_mels_signal_augmented,
             mean_signal_length)







