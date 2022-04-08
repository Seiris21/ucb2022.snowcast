# SnowCast - Berkeley MIDS 2022 Capstone Project
**Authors: Malachy Moran, Derrick Hee, Kayla Wopschall, Tilman Bayer**

![Snowpack Prediction](https://github.com/Seiris21/2022_snowpack_capstone/blob/main/snowcast_prediction_module/ReferenceImages/Tests/Kaweah_2022-03-08%2000:00:00_prediction.png?raw=true)


# Project Background

Accurately estimating the amount of snowfall in the Sierra Nevadas is critically important for the people of California and those who benefit from California agriculture. Over 60% of developed water sources and 75% of agricultural water supply across California are derived from precipitation and snow in the Sierra Nevada (Huning & Margulis 2017). California agriculture accounts for over 35% of the United States vegetables, and over 65% of its fruits and nuts (Pathak et al. 2018), contributing to an agricultural industry valued at over $50.5 billion (CDFA), and employing between 600,000- 800,000 farmworkers in the state (CFF). The Sierra Nevada snowpack also provides more than 15% of the electrical power supply for all of California (Rheinheimer et al. 2012). The implications of water availability in the California Central Valley range widely from food availability throughout the US, to  industry, economy, and the livelihood of Californians.

Here we take a machine learning approach to improving the estimates of snow fall across the Sierra Nevadas by attempting to estimate the Snow Water Equivalent (SWE) for the entire region using satellite and weather data that is readily available on a daily basis. We first give an overview of previous research that has been done regarding predictions of snow measures in the Sierra Nevadas, including previous modeling and other machine learning attempts. We then give an overview of the deep learning models used in our estimations, Convolutional Neural Networks (CNN) and Long short-term memory (LSTM) frameworks,  and the history of their use in other fields. We’ll then introduce the data, methods, and results of our CNN/LSTM modeling efforts for the SWE in the Sierra Nevadas. Lastly, we’ll discuss future improvements and work that can be made to continue working towards an accurate and effective SWE estimation that can assist with the current snowpack and water variability crisis in California.

Our study area centers around nine basins in the Sierra Nevada mountains of California: Feather, Yuba, Truckee, Carson, Tuolumne, Merced, San Joaquin, Kings Canyon, and Kaweah. Selection of these basins was driven by the data available through the National Snow and Ice Data Center (NSIDC) Airborne Snow Observatory (ASO) dataset.

# This Repo

This repository contains the code, working notebooks, and other files that were used in the development of the [snowcast package](https://github.com/Malachyiii/snowcast_package) and the [SnowCast Prediction tool](https://github.com/Seiris21/2022_snowpack_capstone/tree/main/snowcast_prediction_module) which is contained in the snowcast_prediction_module folder.

## File Structure

The most important folders and subfolders of the project are detailed below in the order in which they appear in the repo.

### Data

The data folder contains files relevant to the collection of data, including the data dictionary, and a csv of the training data geometries.

### Docs

The docs folder contains team documents and the presentations of the project, including the [Final Presentation](https://docs.google.com/presentation/d/1Sg37yekpnLrlj9dfBAFUKE6L2FplE75gE6lvkIfVW5k/edit?usp=sharing).

### Models

The models folder contains all the notebooks for models that were trained during this project. The model used to generate the final weights is contained in the [swe_cnn-lstm-dhee-aws](https://github.com/Seiris21/2022_snowpack_capstone/blob/main/models/swe_cnn-lstm-dhee-aws.ipynb) notebook.

### Notebooks

This folder contains all the Jupyter notebooks that were used in the devolpement of this project. One particular folder is of note.

#### notebooks/data_ingestion

The notebooks in the folder labeled data_ingestion show the development of the functions that were used to build the [snowcast package](https://github.com/Malachyiii/snowcast_package). These notebooks are intended to be illustrative only of the work that was done. A thorough and detailed explanation of each function used in data gathering is available in the [snowcast package](https://github.com/Malachyiii/snowcast_package) repo. Please refer to the README.md of that repo.

### Reports

This folder contains the [Final Paper](https://docs.google.com/document/d/1b_gI8lQ0ZhayQcq4T0wT4w9wVk4rD281uSmRsbcSRRc/edit?usp=sharing) for this project. This paper goes into extensive detail on the background, motivation, methodology and results for this project.

### SnowCast Prediction Module

This folder contains the [Final Product](https://github.com/Seiris21/2022_snowpack_capstone/tree/main/snowcast_prediction_module) of this project, a python script which generates predictions of Snow Water Equivalent for the user. This folder contains its own README.md, please refer to this document for a description of the SnowCast tool.

# Wrap-Up

If you have any questions about this project, or wish to receive more information about the project, the contact details for the contributors are below.

Malachy Moran
malachy.j.moran@berkeley.edu

Derrick Hee
dhee@berkeley.edu

Kayla Wopschall
kaylaw@berkeley.edu

Tilman Bayer
tbay@berkeley.edu
