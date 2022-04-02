#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:18:18 2022

@author: malachyiii
"""

print('''
                                                                       
 ,---.                             ,-----.                ,--.   
'   .-' ,--,--,  ,---. ,--.   ,--.'  .--./ ,--,--. ,---.,-'  '-. 
`.  `-. |      \| .-. ||  |.'.|  ||  |    ' ,-.  |(  .-''-.  .-' 
.-'    ||  ||  |' '-' '|   .'.   |'  '--'\\ '-'  |.-'  `) |  |   
`-----' `--''--' `---' '--'   '--' `-----' `--`--'`----'  `--'   

     Please wait while program is initialized...                                                              
'''
      )

######### Start-Up ##############

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

try:
    print("Attempting Package Import")
    from snowcast import data_wrangling
    import ee
    import time
    import numpy as np
    from datetime import datetime
    print("Import Successful")
except ImportError:
    print("Installing Required Package")
    import sys
    import subprocess
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                           'snowcast-Malachyiii'])
    print("Attempting Package Import")
    from snowcast import data_wrangling
    import ee
    import time
    import numpy as np
    from datetime import datetime
    print("Import Successful")


try:
    print("Connecting to Google Earth Engine...")
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()
    
test = input("Do you want to test data importing? [y/n]: ")
if test.lower() == 'y':
    data_wrangling.testing()
    print("Testing Complete")


######## User Input ##########

basin_ref = ["FeatherRef.tif", "YubaRef.tif", "TruckeeRef.tif", "CarsonRef.tif",
             "TuolumneRef.tif", "MercedRef.tif", "SanJoaquinRef.tif", "KingsCanyonRef.tif",
             "KaweahRef.tif"]

valid_input_received = False
while  not valid_input_received:
    basin = input('''Which Basin do you want to generate a prediction for?
    1: Feather
    2: Yuba
    3: Truckee
    4: Carson
    5: Tuolumne
    6: Merced
    7: San Joaquin
    8: Kings Canyon
    9: Kaweah
    Please enter a number: ''')
    
    try:
        if int(basin) in range(1,10):
            valid_input_received = True
        else:
            print("Please enter a number between 1 and 9")
            time.sleep(2)
    except Exception as e:
        print("Please enter a number")
        time.sleep(2)

basin = basin_ref[int(basin)-1]



valid_input_received = False
while  not valid_input_received:
    date = input('''What date would you like to generate a prediction for?
Please enter a date in the dd-mm-yyyy format: ''')
    
    try:
        date = datetime.strptime(date, "%d-%m-%Y")
        valid_input_received = True
    except Exception as e:
        print("Please enter the date in dd-mm-yyyy format, such as 01-01-2020")
        time.sleep(2)



###########DataFrame Generation###############
print("Generating your prediction...")
print("You MIGHT want to go get some coffee...")

tif_path = "ReferenceImages/" + basin


# I expect to see RuntimeWarnings in this block
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    df = data_wrangling.chop_aso(tif_path)
    

##########Define Model Here####################

def model(modis, copernicus, sen1, sen2a, sen2b, weather):
    return 1

##########Prediction module####################

predictions = []

x = 0

totalstart = time.time()

for i in range(len(df)):
    if np.isnan(df.SWE[i]):
        predictions.append(np.nan)
        x+= 1
    else:
        start = time.time()
        MOD10A1 = data_wrangling.pull_MODIS_list(df.geometry[i], date, 'MOD10A1')
        MYD10A1 = data_wrangling.pull_MODIS_list(df.geometry[i], date, 'MYD10A1')
        modis = MOD10A1[2::] + MYD10A1[2::]
        
        copernicus = data_wrangling.get_copernicus(df.geometry[i])
        
        sen1 = data_wrangling.pull_Sentinel1(df.geometry[i], date)
        sen2a = data_wrangling.pull_Sentinel2a(df.geometry[i], date)
        sen2b = data_wrangling.pull_Sentinel2b(df.geometry[i], date)
        weather = data_wrangling.pull_GRIDMET(df.geometry[i], date)
        
        output = model(modis, copernicus, sen1, sen2a, sen2b, weather)
        
        predictions.append(output)
        x+=1
        
        print(f'{round(x/len(df),3)*100}% -- {x} out of {len(df)} km complete')
        print(f"Current time per km: {round(time.time()-start, 3)} seconds")

df.SWE = predictions

print(f"Prediction complete! Total time was {round(time.time()-start, 3)} seconds")
print("Stitching image....")

data_wrangling.stitch_aso(tif_path, df)

print(f'''There is now an image named {basin[0:-7]}_prediction.jpeg in the
snowcast_prediction/ReferenceImages directory''')







