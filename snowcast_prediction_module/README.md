# Prediction Module

The Snowcast prediction module contains the GUI and required files to run it.

# Files

## SnowCast.py

Snowcast.py is a python script intended to be run from terminal. It allows the user to specify any of the 9 basins in the Sierra Nevada mountains covered by [Airborne Snow Observatory (ASO)](https://data.airbornesnowobservatories.com/) flights and a day, then makes a prediction of the Snow Water Equivalent (SWE) for that day.

It can be run from terminal by navigating to this folder on your local drive and running

`python3 SnowCast.py`

### Dependencies

In order to use this tool you must have an account on [Google Earth Engine](https://signup.earthengine.google.com/). This account must be reviewed by google first, but the process is typically fast. You may be asked by SnowCast.py to sign in to this account to facilitate pulling satellite imagery.

SnowCast is heavily dependent on the `snowcast-Malachyiii` package and will try to pip install it if it is not installed already. This package itself has several dependencies, some of which may cause issues. If you are on a Windows machine, you will need to install GDAL first from a wheel. See [snowcast_package](https://github.com/Malachyiii/snowcast_package) for more details.

SnowCast is also heavily dependent on `torch` and `pytorch-lightning`. If you machine has a GPU with cudnn and cudatoolkit installed, it will predict using the GPU. Otherwise it will attempt to use the CPU. This will take much longer and may slow down other processes.

### Timeframe

SnowCast.py takes a significant amount of time to make a single prediction. The timeline ranges from 5-6 hours on GPU for the smallest basin (Kaweah) to 24 hours+ for the largest (Feather). As long as the terminal remains open, the prediction will continue to run.

### Errors

This script is very complex, as it involves downloading data from 7 different sources. Many possible errors have been accounted for, but some errors are beyond our control. If you receive an error from Google Earth Engine or Microsoft Planetary Computer, often the best option is to wait an hour or two and try again. If you consistently receive the same errors please email malachy.j.moran@gmail.com.

### Output

Snowcast.py will output 4 files to the ReferenceImages directory that it is stored in.
1. A .png heatmap of predictions with the filename structure "{basin}_{datetime}_predictions.png"
2. An identically named .csv file of these predictions
3. A .png heatmap of smoothed predictions with the filename structure "{basin}_{date}_smoothed_predictions.png"
4. An identically named .csv file of these smoothed predictions

## scaler.pkl

This pickle file is required for the model to return valid predictions. It contains the weights for the Min/Max scaler used during model training.

## ReferenceImages directory

This directory contains .tiff files that define the basin outlines. These files have the filename structure "{basin}Ref.tiff" If any of these files are deleted, SnowCast.py will be unable to make predictions for that basin.

### Output Files

This is the folder where all files output by SnowCast.py will land.

## snowcast_best_weights.ckpt

This file will not exist when you first clone the directory, but will be downloaded from a public google drive [file](https://drive.google.com/file/d/1-Zk60aN6ImQ4XResdTvKeriP_DlTGofu/view?usp=sharing) the first time the program is run.

# Wrap-Up

This module will likely be refined and updated as time goes on. Please keep an eye on the github for changes and updates that may arise
