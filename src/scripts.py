def get_copernicus(traindf, path, overwrite = False):
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

