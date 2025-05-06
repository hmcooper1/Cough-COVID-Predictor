#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=1000gb
#PBS -N 1-pitch_shift

cd /rds/general/project/hda_24-25/live/ML/Group14/data_augmentation

module load anaconda3/personal
source activate test1

python 1-pitch_shift.py