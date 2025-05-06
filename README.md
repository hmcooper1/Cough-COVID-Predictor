# Cough-COVID-Predictor
## Overview
This project uses the (COUGHVID dataset)[https://zenodo.org/records/4498364#.Yi8m2RDP1MD] to detect COVID-19 from cough audio recordings. The dataset consists of 25,000+ crowdsourced cough audio samples collected between April 1st, 2020 and December 1st, 2020 through a web application. Each sample is accompanied by self-reported metadata, including COVID-19 status, presence of respiratory symptoms, and dmeographic information. Melspectograms were used to transform the raw audio signals into an image that is suitable input for convolutional neural networks (CNN). The goal is to develop a model capable of identifying COVID-19 infections from cough audio alone, enabling a low-cost, accessible, and non-invasive method for preliminary screen that could be deployed through mobile devices or web-based platforms.
## Data Preprocessing
- data_cleaning.ipynb: basic data cleaning from metadata
- data_augmentation/1-pitch-shift.py: address class imbalance and enrich dataset
  - For COVID-19 samples: save original audio, use PitchShifting (lower the original pitch of a sound) from Librosa to shift audio samples down 4 steps
  - For healthy samples: only save original audio
- data_augmentation/2-melspectogram_augment.py: address class imbalance, enrich dataset, turn raw audio samples into melspectograms
  - For COVID-19 samples: 
## Models

## Results

## Reference?
