---
layout: page
title: Solution and Results
permalink: /results/
---

# Our Solution
Utilizing publicly available datasets, this project aimed to create a tool that would be able to provide accurate estimates of the Snow Water Equivalent (SWE) for 9 large basins in the Sierra Nevada Mountains at a 1km resolution. 

By using public datasets as inputs for our model, we aim to replicate costly surveying techniques at little to no cost. Our hope is that the tool we created can be utilized by environmental scientists and water planners to study the snowpack and steward California's water supply into the future.

# Our Tool

![Pipeline](https://raw.githubusercontent.com/Seiris21/ucb2022.snowcast/main/docs/assets/pipeline.png)

## Installation and Usage:
Setting up and running the Snowcast tool is incredibly easy, with only 5 steps to making your first prediction

Steps:
1. Clone the [git repo](https://github.com/Seiris21/ucb2022.snowcast)
2. Download the [python package](https://github.com/Malachyiii/snowcast_package) with `pip install snowcast-Malachyiii`
3. Open a terminal and navigate to the prediction module subfolder
4. Run the tool with `python3 SnowCast.py`
5. Enter the basin and date you want to make a prediction for

That's all there is to it! Watch the short installation guide and demo below to see a demonstration.


<iframe width="560" height="315" src="https://www.youtube.com/embed/CwJyJ6Lwvjg" title="Snowcast Tutorial" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Tool Datasets:
The Snowcast prediction tool utilizes three different types of datasets.

1. Airborne flyover data from the Airborne Snow Observatory was utilized as the target variable
2. Satellite imagery from the Sentinel, MODIS, and Copernicus datasets
3. Weather data obtained from GRIDMET

All of these datasets can be accessed below, and are available on Google Earth Engine or Microsoft Planetary Computer

### Target Variable
- [Airborne Snow Observatory](https://nsidc.org/data/aso)

### Satellite Imagery and Weather Assets
- [Sentinel-1 SAR GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- [Sentinel-2 MSI](https://developers.google.com/earth-engine/datasets/catalog/sentinel-2)
- [MODIS Terra and Aqua Snowcover Data](https://developers.google.com/earth-engine/datasets/catalog/modis)
- [Copernicus 30m Digital Elevation Model](link)
- [GRIDMET Meteorogical Dataset](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET)



# Model Results and Accuracy Measures

![Smoothed Prediction of Yuba Basin for March 25th, 2022](https://raw.githubusercontent.com/Seiris21/ucb2022.snowcast/main/snowcast_prediction_module/ReferenceImages/Tests/Yuba_2022-03-25 00:00:00_smoothed_prediction.png)


### Table of Results by Basin

| Basins (N-S) | Basin Size (km^2) | RMSE Observed | Smoothed RMSE Observed |
| --- | --- | --- | --- | --- |
| Feather | 8371 | 2.733" | 2.686" |
| Yuba | 2203 | 7.626" | 7.010" |
| Truckee | 2915 | 10.456" | 9.442" |
| Carson | 1478 | 7.710" | 7.343" |
| Tuolumne | 2921 | 9.501" | 9.497" |
| Merced | 1711 | 6.888" | 5.751" |
| San Joaquin | 4242 | 8.837" | 7.637" |
| Kings Canyon | 3464 | 18.411" | 17.485" |
| Kaweah | 1451" | 5.300" | 5.120" |
| Overall | 28756 | 8.000" | 7.452" |
