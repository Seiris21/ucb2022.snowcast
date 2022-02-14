"""
Scripts used to pull satellite images to supplement model
"""
import os
import planetary_computer
import requests
import shutil
import xarray
import xrspatial

from datetime import timedelta
from pystac_client import Client


def pull_MODIS(traindf, modis, path, overwrite = False, names_only = False):
  """
  dataframe: traindf - pandas/geopandas dataframe with date and geometry columns
  str: modis - one of ["MOD10A1","MYD10A1"]
  str: path - path to write images to
  bool: overwrite - overwrite existing images in path
  bool: names_only - return filenames only
  """
  filelocations = []
  x = 0

  for i in range(len(traindf.SWE)):

    #create a name for the image
    pict_name = traindf.cell_id[i] + '_' + modis + '_' + traindf.datestring[i] + '.jpg'

    #create the whole filename with path to the correct folder
    filename = os.path.join(path, modis, pict_name)

    if names_only:
      filelocations.append(filename)
      x += 1
      if x % 5000 == 0:
        print(f'{x} files already exist')

    elif os.path.exists(filename) and not overwrite:
      filelocations.append(filename)
      x += 1
      if x % 5000 == 0:
        print(f'{x} files already exist')

    else:
      #We need a start date and an end date. Just like a regular python slice, 
      #the end date is not included, so by using a 1 day frame, I am actually limiting
      #the range to only the day in question
      start_date = traindf.date[i] - timedelta(days = 7)
      end_date = traindf.date[i] + timedelta(days = 1)

      #First I get the image collection from the MODIS data, filter it only to the days in question
      #and select my bands, then sort so the most recent day in the group is at the top
      Collection = ee.ImageCollection(f'MODIS/006/{modis}') \
                  .filter(ee.Filter.date(start_date, end_date)) \
                  .filter(ee.Filter.notNull(['system:index'])) \
                  .select(['NDSI_Snow_Cover', 'Snow_Albedo_Daily_Tile', 'NDSI']) \
                  .sort('system:index', False) 

      #I create a google earth images point based on the area centroid
      centroid = ee.Geometry.Point(traindf.center_long[i], traindf.center_lat[i])

      #Because the image collection is limited to a single day, there is only one image
      #So I just take it
      point = Collection.first().unmask(0)

      # Get individual band arrays and build them into an RGB image
      # The "buffer" is a circular distance around the point, measured in meters right now it is 100km
      rgb = ee.Image.rgb(point.clip(centroid.buffer(10000)).select('NDSI_Snow_Cover').divide(100), #I divide by 100 to get it between 0 and 1
                        point.clip(centroid.buffer(10000)).select('Snow_Albedo_Daily_Tile').divide(100), #I divide by 100 to get it between 0 and 1
                        point.clip(centroid.buffer(10000)).select('NDSI').divide(10000)).visualize() #I divide by 10000 to get it between 0 and 1

      #Now I get the url for the image
      url = rgb.getThumbURL()

      #add the name to my list I created earlier
      filelocations.append(filename)

      #now I open the url and download the image to the specified file location
      response = requests.get(url, stream=True)
      with open(filename, 'wb') as out_file:
          shutil.copyfileobj(response.raw, out_file)
      del response
    
  traindf[f"{modis}_filelocations"] = filelocations


def get_copernicus(traindf, path, overwrite = False):
  """
  dataframe: traindf - pandas/geopandas dataframe with geometry columns
  str: path - path to write images to
  bool: overwrite - overwrite existing images in path
  """
  traindf["copernicus_filelocations"] = "blank"
  x = 0
  length_cell_id = len(traindf.cell_id.unique())

  for i in traindf.cell_id.unique():
    #create a name for the image
    pict_name = i + '_' + 'copernicus90m'

    #create the whole filename with path to the correct folder
    filename = os.path.join(path, pict_name)

    # Adapted from https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-90#Example-Notebook :
    
    if not os.path.exists(filename + '.png') or overwrite:
      client = Client.open(
          "https://planetarycomputer.microsoft.com/api/stac/v1",
          ignore_conformance=True,
      )

      point = [traindf.loc[traindf.cell_id == i, "center_long"].iloc[0], 
              traindf.loc[traindf.cell_id == i, "center_lat"].iloc[0]]
      
      search = client.search(
          collections=["cop-dem-glo-90"],
          intersects={"type": "Point", "coordinates": point},
      )

      items = list(search.get_items())

      signed_asset = planetary_computer.sign(items[0].assets["data"])
      
      data = (
          xarray.open_rasterio(signed_asset.href)
          .squeeze()
          .drop("band")
          .mean()
      )
      min_lon = min([j[0] for j in [y for y in traindf.loc[traindf.cell_id == i, "geometry"].iloc[0].centroid.buffer(.05).boundary.coords]])
      min_lat = min([j[1] for j in [y for y in traindf.loc[traindf.cell_id == i, "geometry"].iloc[0].centroid.buffer(.05).boundary.coords]])
      max_lon = max([j[0] for j in [y for y in traindf.loc[traindf.cell_id == i, "geometry"].iloc[0].centroid.buffer(.05).boundary.coords]])
      max_lat = max([j[1] for j in [y for y in traindf.loc[traindf.cell_id == i, "geometry"].iloc[0].centroid.buffer(.05).boundary.coords]])

      mask_lon = (data.x >= min_lon) & (data.x <= max_lon)
      mask_lat = (data.y >= min_lat) & (data.x <= max_lat)

      cropped_data = data.where(mask_lon & mask_lat, drop=True)

      hillshade = xrspatial.hillshade(cropped_data)
      img = stack(shade(hillshade, cmap=["white", "gray"]), shade(cropped_data, cmap=Elevation, alpha=128))
      export_image(img=img, filename=filename, background=None)

    traindf.loc[traindf.cell_id == i, "copernicus_filelocations"] = filename + '.png'
    if x % 500 == 0:
      print(f'{x} out of {length_cell_id} complete')
    x += 1

