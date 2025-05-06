# Cough-COVID-Predictor
## Overview
This project uses the [COUGHVID dataset](https://zenodo.org/records/4498364#.Yi8m2RDP1MD) to detect COVID-19 from cough audio recordings. The dataset consists of 25,000+ crowdsourced cough audio samples collected between April 1st, 2020 and December 1st, 2020 through a web application. Each sample is accompanied by self-reported metadata, including COVID-19 status, presence of respiratory symptoms, and dmeographic information. Melspectograms were used to transform the raw audio signals into an image that is suitable input for convolutional neural networks (CNN). The goal is to develop a model capable of identifying COVID-19 infections from cough audio alone, enabling a low-cost, accessible, and non-invasive method for preliminary screen that could be deployed through mobile devices or web-based platforms.
## Data Preprocessing
- data_cleaning.ipynb: basic data cleaning from metadata
- data_augmentation/1-pitch-shift.py: address class imbalance and enrich dataset
  - For COVID-19 samples: save original audio, use PitchShifting (lower the original pitch of a sound) from Librosa to shift audio samples down 4 steps
  - For healthy samples: only save original audio
- data_augmentation/2-melspectogram_augment.py: address class imbalance, enrich dataset, turn raw audio samples into melspectograms
  - For COVID-19 samples: generate one unaugmented spectogram and two augmented versions through frequency and time masking
  - For healthy samples: generate one unaugmented spectogram and one augmented version through frequency and time masking

Code in data_augmentation folder adapted from [skanderhamdi/attention_cnn_lstm_covid_mel_spectrogram](https://github.com/skanderhamdi/attention_cnn_lstm_covid_mel_spectrogram), with modifications for our project needs.
## Models
- cnn.ipynb: CNN model trained on melspectogram images for COVID-19 classification
- cnn_lstm.ipynb: hybrid CNN-LSTM model that first extracts spatial features from melspectograms using a CNN, then feeds temporal sequence of features into an LSTM to capture time-dependent patterns for COVID-19 classification
## Results

## References
This [paper](https://link.springer.com/article/10.1007/s10844-022-00707-7) outlines the data proprocessing and augmentation techniques used in our approach. 
