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

This folder contains all the Jupyter Notebooks that were used in the devolpement of this project. One particular folder is of note.

#### notebooks/data_ingestion

The notebooks in the folder labeled data_ingestion show the development of the functions that were used to build the [snowcast package](https://github.com/Malachyiii/snowcast_package). These notebooks are intended to be illustrative only of the work that was done. A thorough and detailed explanation of each function used in data gathering is available in the [snowcast package](https://github.com/Malachyiii/snowcast_package) repo. Please refer to the README.md of that repo.

### Reports

This folder contains the [Final Paper](https://docs.google.com/document/d/1b_gI8lQ0ZhayQcq4T0wT4w9wVk4rD281uSmRsbcSRRc/edit?usp=sharing) for this project. This paper goes into extensive detail on the background, motivation, methodology and results for this project.

### SnowCast Prediction Module

This folder contains the [Final Product](https://github.com/Seiris21/2022_snowpack_capstone/tree/main/snowcast_prediction_module) of this project, a python script which generates predictions of Snow Water Equivalent for the user. This folder contains it's own README.md, please refer to this document for a discription of the SnowCast tool.

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


# References

REFERENCES: 

Aschauer, J., & Marty, C. (2021). Evaluating methods for reconstructing large gaps in historic snow depth time series. Geoscientific Instrumentation, Methods and Data Systems, 10(2), 297-312.

Behrangi, Ali, et al. “Using the Airborne Snow Observatory to Assess Remotely Sensed Snowfall Products in the California Sierra Nevada.” Water Resources Research, vol. 54, no. 10, Oct. 2018, pp. 7331–46. https://doi.org/10.1029/2018WR023108  .

Berg, N., & Hall, A. (2017). Anthropogenic warming impacts on California snowpack during drought. Geophysical Research Letters, 44(5), 2511-2518.

CDFA (California Department of Food and Agriculture). California Agricultural Statistics Review 2015–2016. Available online: https://www.cdfa.ca.gov/statistics/PDFs/2016Report.pdf (accessed on 29 March 2022).

CFF, Center for Farmworker Families. https://farmworkerfamily.org/, (accessed on 2 April 2022).

Cayan, D. R., Maurer, E. P., Dettinger, M. D., Tyree, M., & Hayhoe, K. (2008). Climate change scenarios for the California region. Climatic change, 87(1), 21-42.

Chang, H., & Bonnette, M. R. (2016). Climate change and water‐related ecosystem services: impacts of drought in California, USA. Ecosystem Health and Sustainability, 2(12), e01254.

Dettinger, M. D., & Anderson, M. L. (2015). Storage in California's reservoirs and snowpack in this time of drought. San Francisco Estuary and Watershed Science, 13(2).

Dettinger, M.D., Redmond, K., & Cayan, D. (2004). Winter orographic precipitation ratios in the Sierra Nevada—Large-scale atmospheric circulations and hydrologic consequences. Journal of Hydrometeorology, 5(6), 1102-1116.

Duan, S., & Ullrich, P. (2021). A comprehensive investigation of machine learning models for estimating daily snow water equivalent over the Western US.

Harte, J., Saleska, S. R., & Levy, C. (2015). Convergent ecosystem responses to 23‐year ambient and manipulated warming link advancing snowmelt and shrub encroachment to transient and long‐term climate–soil carbon feedback. Global Change Biology, 21(6), 2349-2356.

Huning, L. S., & Margulis, S. A. (2018). Investigating the variability of high-elevation seasonal orographic snowfall enhancement and its drivers across Sierra Nevada, California. Journal of Hydrometeorology, 19(1), 47-67.

Huning, L. S., & Margulis, S. A. (2017). Climatology of seasonal snowfall accumulation across the Sierra Nevada (USA): Accumulation rates, distributions, and variability. Water Resources Research, 53(7), 6033-6049.

Huning, L. S., Margulis, S. A., Guan, B., Waliser, D. E., & Neiman, P. J. (2017). Implications of detection methods on characterizing atmospheric river contribution to seasonal snowfall across Sierra Nevada, USA. Geophysical Research Letters, 44(20), 10-445.

Kirchner, P. B., Bales, R. C., Molotch, N. P., Flanagan, J., & Guo, Q. (2014). LiDAR measurement of seasonal snow accumulation along an elevation gradient in the southern Sierra Nevada, California. Hydrology and Earth System Sciences, 18(10), 4261-4275.

Lehning, M. (2013). Snow-atmosphere interactions and hydrological consequences. Advances in Water Resources, 55, 1-3.

Lundquist, J. D., Minder, J. R., Neiman, P. J., & Sukovich, E. (2010). Relationships between barrier jet heights, orographic precipitation gradients, and streamflow in the northern Sierra Nevada. Journal of Hydrometeorology, 11(5), 1141-1156.

Margulis, S. A., Cortés, G., Girotto, M., & Durand, M. (2016). A Landsat-era Sierra Nevada snow reanalysis (1985–2015). Journal of Hydrometeorology, 17(4), 1203-1221.

Margulis, S. A., Girotto, M., Cortés, G., & Durand, M. (2015). A particle batch smoother approach to snow water equivalent estimation. Journal of Hydrometeorology, 16(4), 1752-1772.

Meyal, A. Y., Versteeg, R., Alper, E., Johnson, D., Rodzianko, A., Franklin, M., & Wainwright, H. (2020). Automated Cloud Based Long Short-Term Memory Neural Network Based SWE Prediction. Frontiers in Water, 53.

Pile, L. S., Meyer, M. D., Rojas, R., Roe, O., & Smith, M. T. (2019). Drought impacts and compounding mortality on forest trees in the southern Sierra Nevada. Forests, 10(3), 237.

Prugh, L. R., Deguines, N., Grinath, J. B., Suding, K. N., Bean, W. T., Stafford, R., & Brashares, J. S. (2018). Ecological winners and losers of extreme drought in California. Nature Climate Change, 8(9), 819-824.

Rheinheimer, D. E., Ligare, S. T., & Viers, J. H. (2012). Water and Energy Sector Vulnerability to Climate Warming in the Sierra Nevada: Simulating the Regulated Rivers of California’s West Slope Sierra Nevada.

Sloat, L. L., Henderson, A. N., Lamanna, C., & Enquist, B. J. (2015). The effect of the foresummer drought on carbon exchange in subalpine meadows. Ecosystems, 18(3), 533-545.

Storey, E. A., Stow, D. A., Roberts, D. A., O’Leary, J. F., & Davis, F. W. (2021). Evaluating drought impact on postfire recovery of chaparral across southern California. Ecosystems, 24(4), 806-824.

Wainwright, H. M., Steefel, C., Trutner, S. D., Henderson, A. N., Nikolopoulos, E. I., Wilmer, C. F., ... & Enquist, B. J. (2020). Satellite-derived foresummer drought sensitivity of plant productivity in Rocky Mountain headwater catchments: spatial heterogeneity and geological-geomorphological control. Environmental Research Letters, 15(8), 084018.
