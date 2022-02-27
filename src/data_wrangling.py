'''
Functions used for dataset wrangling

'''

import os
import re
import numpy as np

import geopandas as gpd
import pandas as pd


from sklearn.neighbors import BallTree

#Adapted from AutoGIS| University of Helsinki
# https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
def get_knearest(src_points, candidates, knn=1):
    '''
    K nearest neighbors for every source point given candidate points
    '''
    #Make candidates BallTree format
    tree = BallTree(candidates,leaf_size=15,metric='haversine')

    #Find closest points
    distances, indices = tree.query(src_points, k=knn)

    #Transpose into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    return(indices, distances)

def nearest_neighbor(left_gdf, right_gdf, return_dist=False, knn=1):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    #Some Nan buffer to KNN search
    knn = knn*3

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    # For left radians, data is in polygon format, so apply meter crs, get centroid, and revert
    left_radians = np.array(left_gdf[left_geom_col].to_crs('epsg:4087').centroid.to_crs("EPSG:4326").apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())


    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_knearest(src_points=left_radians, candidates=right_radians, knn=knn)

    #return(closest,dist)

    closest_points = gpd.GeoDataFrame()

    #Loop for knn
    for i in range(knn):
    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    #Loop to return closest starting from 0 idx
    closest_points['station_id_'+str(i)] = right['station_id'].loc[closest[i]].values
    closest_points['elevation_m_'+str(i)] = right['elevation_m'].loc[closest[i]].values

    # Add distance if requested
    if return_dist:
      # Convert to meters from radians
      earth_radius = 6371000  # meters
      closest_points['distance_'+str(i)] = dist[i] * earth_radius

    return closest_points

def inverseDmean(df,power):
    #Formula for inverse Distance Average = ((x1/d1^p)+(x2/d2^p)....)/((1/d1^p)+(1/d2^p)....)
    #https://gisgeography.com/inverse-distance-weighting-idw-interpolation/
    subset = df.filter(regex='distance_[0-9]+|SWE_[0-9]+')
    numerator = pd.DataFrame()
    denominator = pd.DataFrame()

    for i in range(int(subset.shape[1]/2)):
    numerator['x_'+str(i)]=subset['SWE_'+str(i)]/(subset['distance_'+str(i)]**power)
    denominator['x_'+str(i)]=1/(subset['distance_'+str(i)]**power)

    #There are cells without SWE data. We do not want this in the inverse Distance Calculation
    nulls = np.where(pd.isnull(numerator))
    for row,column in zip(nulls[0],nulls[1]):
    denominator.at[row, denominator.columns[column]] = np.nan

    numerator['sum']=numerator.sum(axis=1)
    #print(numerator.head())
    denominator['sum']=denominator.sum(axis=1)
    #print(denominator.head())

    return(numerator['sum']/denominator['sum'])
###End of adapted functions


def swe_calculation(train, labels, closest_stations, knn=1):
    #Join labels with closest_stations
    labels_joined = labels.join(closest_stations)

    #Prepare column names
    SWE_names=[]
    elevation_names=[]
    reordered_columns = ['cell_id', 'date', 'SWE', 'region', 'geometry',
                       'mean_inversed_swe', 'mean_local_swe',	'median_local_swe',	'max_local_swe', 'min_local_swe',
                       'mean_local_elevation',	'median_local_elevation',	'max_local_elevation','min_local_elevation']
    for i in range(knn):
    reordered_columns.extend(['station_id_'+str(i),'elevation_m_'+str(i),'distance_'+str(i),'SWE_'+str(i)])
    SWE_names.append('SWE_'+str(i))
    elevation_names.append('elevation_m_'+str(i))

    #Merge against cell_id+date to get closest stations for each cell
    idx = 0
    for i in range(knn*3):
    train
    if i == 0:
      tmp_merged = pd.merge(labels_joined, train, how="left", left_on=['station_id_'+str(i), 'date'], right_on=['station_id','date'],suffixes=(None,'_'+str(i))).drop(columns= ['station_id'])
    else:
      tmp_merged = pd.merge(tmp_merged, train, how="left", left_on=['station_id_'+str(i), 'date'], right_on=['station_id','date'],suffixes=(None,'_'+str(i))).drop(columns= ['station_id'])

    #Filter out nearest neighbors with NaN, get 5 closest WITH VALUES
    filtered = []
    for idx,row in tmp_merged.iterrows():
    index = []
    values = []
    i=0
    counter=0
    while i<knn:
      if not pd.isna(row['SWE_'+str(counter)]):
        i+=1
        index.append(counter)
      counter+=1
    for j in index:
      values.extend([row['station_id_'+str(j)], row['elevation_m_'+str(j)], row['distance_'+str(j)], row['SWE_'+str(j)]])
    filtered.append(values)

    #Re-merge with cell data
    merged_train = labels.join(pd.DataFrame(filtered,columns=reordered_columns[-4*knn:]))

    #Calculations
    #Elevations
    # Normal Mean
    merged_train['mean_local_elevation']=merged_train[elevation_names].mean(axis=1)
    # Median
    merged_train['median_local_elevation']=merged_train[elevation_names].median(axis=1)
    # Max
    merged_train['max_local_elevation']=merged_train[elevation_names].max(axis=1)
    # Min
    merged_train['min_local_elevation']=merged_train[elevation_names].min(axis=1)

    #SWE
    #Inverse Distance Mean
    merged_train['mean_inversed_swe']=inverseDmean(merged_train,2)
    #Normal Mean
    merged_train['mean_local_swe']=merged_train[SWE_names].mean(axis=1)
    #Median
    merged_train['median_local_swe']=merged_train[SWE_names].median(axis=1)
    #Min
    merged_train['min_local_swe']=merged_train[SWE_names].min(axis=1)
    #Max
    merged_train['max_local_swe']=merged_train[SWE_names].max(axis=1)

    #Reorder Columns
    merged_train=merged_train[reordered_columns]

    return(merged_train)

def rawdata_merge(metadata,features,labels, geojson,knn=5):
    '''
    Takes raw csv data files (From snowcap competition, SNOTEL/CDEC) and merges them for later use.
    Additionally pairs cell_ids with knn station ids.
    
    input:
    pd.dataframe: metadata - dataframe object from station csv. Needs column 'station_id'
    pd.dataframe: features - dataframe object from station csv. Needs column 'station_id'
    pd.dataframe: labels - Ground truth SWE values from cells with cell_id metadata
    gpd.dataframe: geojson - location metadata in geopandas (POLYGON)
    
    output:
    gdp.dataframe: merged - combined dataframe
    '''
    
    #Map Lat/Long to correct crs
    gdf = gpd.GeoDataFrame(metadata, 
                       geometry = gpd.points_from_xy(trainmeta.longitude, trainmeta.latitude),
                       crs = "EPSG:4326")
    #Merge metadata with features in preparation for pairing
    features = features.merge(metadata, how = 'left', on='station_id')


    
    #Merge cell data (trainingset) with location metadata
    labels = labels.melt(id_vars=["cell_id"]).dropna().reset_index(drop = True)
    labels.rename(columns = {"cell_id":"cell_id", "variable":"date", "value":"SWE"}, inplace = True)
    labels = labels.merge(geojson, how = 'left', on='cell_id')
    labels = gpd.GeoDataFrame(traindf, crs ="EPSG:4326")
    
    #Calculate knn stations to get SWE Estimate
    closest_stations = nearest_neighbor(labels, gdf, return_dist=True,knn=knn)
    traindf = swe_calculation(train=features, labels=labels, closest_stations=closest_stations, knn=knn)
    
    #Keep specific columns
    traindf = traindf[['cell_id','date','SWE','region','geometry','mean_inversed_swe',
                   'mean_local_swe','median_local_swe','max_local_swe','min_local_swe',
                   'mean_local_elevation','median_local_elevation','max_local_elevation','min_local_elevation',]]
    
    return(traindf)


def linear_interpolation(trainfeatures,nan_days):
    '''
    Uses linear interpolation to fill in values in dataframe trainfeatures based on missing days
    
    Input:
    pd.dataframe: train features - dataframe of input data. Must include column SWE which will be interpolated
    list: nan_days - list of days in YYYY-MM-DD string format
    
    Output
    list: supplement - list of lists [station_id, date, interpolated swe]
    
    Can be merged onto trainfeatures:
    trainfeatures = trainfeatures.append(pd.DataFrame(supplement, columns ['station_id','date','SWE']))
                                         .sort_values(by='date')
                                         .reset_index(drop=True)
    '''
    #Linear regression implementation, filling in based on nan_dates list
    supplement = []

    #Iterate through all the unique stations
    for station in trainfeatures['station_id'].unique(): #['CDEC:SSM']: #
      #Get subset for this station
      subset = trainfeatures[trainfeatures['station_id']==station].copy()
      #make filler rows with missing dates
      filler = [[station,date,np.nan] for date in nan_dates]
      #Append filler rows to subset and sort on date and reset index
      subset = subset.append(pd.DataFrame(filler, columns=['station_id','date','SWE'])).sort_values(by='date').reset_index(drop=True)
      #print(station,len(subset.index))
      #print(subset.head())
      for date in nan_dates:
        #Find NaN date
        nan_index = subset.index[subset['date'] == date].tolist()[0]
        nan_date = datetime.strptime(date,'%Y-%m-%d')

        #There is a conditional needed for stations that stopped reporting before 2019
        try:
          count=0
          #Find older date that HAS value. Sometimes needed because filler inserted NaNs
          while subset.iloc[nan_index-1-count].isnull().any():
            count+=1
          #Older date (nan-1)
          if (nan_index-1-count)>=0:
            older_date = datetime.strptime(subset.iloc[nan_index-1-count]['date'],'%Y-%m-%d')
            older_swe = subset.iloc[nan_index-1-count]['SWE']
          else:
            older_date = datetime.strptime(subset.iloc[nan_index-1]['date'],'%Y-%m-%d')
            older_swe = np.nan
          #print('Older',nan_index-1,older_date,older_swe)
          #print('NaN-inserted',nan_index,nan_date)

          #Newer date is next date that HAS value, otherwise enter except
          counter=0
          while subset.iloc[nan_index+1+counter].isnull().any():
            counter+=1
          #Newer date
          newer_date = datetime.strptime(subset.iloc[nan_index+1+counter]['date'],'%Y-%m-%d')
          newer_swe = subset.iloc[nan_index+1+counter]['SWE']
          #print('newer',nan_index+1+counter,newer_date,newer_swe)
          #print('______________________________')

          #Change per day
          delta_day = (newer_swe-older_swe)/(newer_date-older_date).days

          #Add expected change to older swe
          est_swe = older_swe + (delta_day*(nan_date-older_date).days)

          #Add "entry" row to supplement
          supplement.append([station,date,est_swe])
        #IndexError happens when the last date is actually from the nan list. Because of this, We DEFINITELY need to do some inter-station interpolation
        except IndexError:
          supplement.append([station,date,np.nan])
    return(supplement)







