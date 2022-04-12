# Predicting Snowpack in California’s Sierra Nevada Mountains

[__Meet the Team__](about_us.md)
[__Solution and Results__](solution_results.md)

## Web Deliverable requirements
Requirements from Week 1 PPT:  

The final web deliverable is a chance to showcase your project,
not just for peers and instructors but also for a broader
audience. Think of it as a portfolio. 

At minimum, the web deliverable should: 
- describe the problem
- model results
- evaluation
- impact
- include a biography or group intro page.   
Interfaces that allow user interaction when appropriate are encouraged, for example, to explore different data sets or parameter choices.

### Problem
Estimation of snowfall in the Sierra Nevadas is critially important for the people of California and those who benefit from California agriculture. 75% of agricultural water supply across California are derived from precipitation and snow in the Sierra Nevadas.  
Current methods to estimate snow water equivalent (SWE) are either biased or expensive to conduct. Snow Telemetry (SNOTEL) sites are an automated network of snowpack and related climate sensors. Although there is constant data due to it's automated nature, the distribution of SNOTEL sites are biased towards lower elevations and do not accurately reflect all areas of interest. Airborne Snow Observatory (ASO) is a program to measure SWE through Light Detection and Ranging (lidar) and is the current gold standard for SWE estimations with it's ability to map entire regions through flyby missions. However ASO data collection is costly, and in recent years have gone private.  
Accurate estimates of snowfall would provide a valuable tool for natural resource managements, allowing for better usage recommendations and minimizing negative impacts for the agricultural region.

### Our Solution
Utilizing publicly available datasets, this project aimed to create a tool that would be able to provide accurate SWE estimates for basins in the Sierra Nevada. ASO was used as our SWE target for training. Public datasets were utilized as inputs for our model. We chose data sources that are going to be available for years to come (satillite imagery, weather data), in the hopes that the tool we created can be continued to be utilized in the future.

![Pipeline](https://raw.githubusercontent.com/Seiris21/ucb2022.snowcast/ac0228f8296cc378b41d5b2340cfbe41e4d0ffb5/docs/assets/pipeline.png)


### Datasets:
- Airborne Snow Observatory (link)
- Sentinel (link)
- MODIS (link)
- Copernicus (link)
- GRIDMET (link)

### Usage:
Steps:
- Manual steps, from video or something


<iframe width="560" height="315" src="https://www.youtube.com/embed/CwJyJ6Lwvjg" title="Snowcast Tutorial" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Model Results
\<Insert images here>

![Test Image](https://raw.githubusercontent.com/Seiris21/ucb2022.snowcast/main/docs/assets/labeled.png)

\<Insert results here>
#### Placeholder (Table from paper)

| Basins (N-S) | Basin Size (km^2) | RMSE Observed | RMSE Predicted | Error |
| --- | --- | --- | --- | --- |
| Feather | - | - | - | - |
| Yuba | - | - | - | - |
| Truckee | - | - | - | - |
| Carson | - | - | - | - |
| Tuolumne | - | - | - | - |
| Merced | - | - | - | - |
| San Joaquin | - | - | - | - |
| Kings Canyon | - | - | - | - |
| Kaweah | - | - | - | - |
| Overall | - | - | - | - |

### [Biography](biography.md)
