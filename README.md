# SnowCast - Berkeley MIDS 2022 Capstone Project
**Authors: Malachy Moran, Derrick Hee, Kayla Wopschall, Tilman Bayer**

![Snowpack Prediction](https://github.com/Seiris21/2022_snowpack_capstone/blob/main/snowcast_prediction_module/ReferenceImages/Tests/Kaweah_2022-03-08%2000:00:00_prediction.png?raw=true)


# Project Background
**Predicting Snowpack in California’s Sierra Nevada Mountains**

California's Central Valley agriculture region accounts for over 35% of all US vegetable production, and over 65% of all US fruit and nut production, in addition to employing over 600k workers, while remaining dependent on snowmelt water from the Sierra Nevada mountains in a time of decreasing water supply. We’ve leveraged CNN and LSTM modeling to create a tool that uses satellite and weather data to estimate peak snow water equivalent (SWE) levels - i.e. amount of water after snow melt - in the Sierra Nevada.

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
