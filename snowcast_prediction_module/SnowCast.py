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
    import pandas as pd
    from datetime import datetime
    from PIL import Image
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
    import pandas as pd
    from datetime import datetime
    from PIL import Image
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
             "TuolumneRef.tif", "MercedRef.tif", "SanJoaquinRef.tif", "KingsRef.tif",
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
print("Generating Basin Reference...")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    df = data_wrangling.chop_aso(tif_path)

print("References Generated")

##########Define Model Here####################
print("Attempting additional imports...")

try: 
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    from torchvision import transforms as T
    
    from torchmetrics import R2Score
    
    import timm
    
    
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning.core.lightning import LightningModule
    
    from scipy import ndimage
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
except ImportError:
    print("Oops! Looks like you're missing some required modeling packages")
    time.sleep(2)
    import sys
    import subprocess
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                           'torch', 'torchvision', 'torchmetrics', 'timm', 'pytorch-lightning',
                           'scipy', 'matplotlib', 'seaborn'])
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    from torchvision import transforms as T
    
    from torchmetrics import R2Score
    
    import timm
    
    
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning.core.lightning import LightningModule
    
    from scipy import ndimage
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

print("Imports Successful")

class args:

    #Keep track of features used in wandb
    lstm_features = ['precip_daily','wind_dir_avg','temp_min','temp_max','wind_vel']

    #Setting the number of CPU workers we are using
    num_workers = 12

    #Setting the seed so we can replicate
    seed = 1212

    #Toggle for whether or not we want our model pretrained on imagenet
    pretrained = True

    #Next we pick the model name with the appropriate shape, img size and output
    model_name1 = 'mixnet_s'
    model_shape1 = 1536
    model_name2 = 'tf_efficientnet_b2_ns'
    model_shape2 = 1408 #768 for swin small 1536 for swin large 1792 for efficientnet b4 768 for cait-m-36
    imagesize = 224
    num_classes = 1
    img_channels = 3

    #LSTM variables
    lstm_hidden = 128
    lstm_layers = 2
    lstm_seqlen = 10

    #Training Args
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1

    #Max epochs and number of folds
    max_epochs = 100
    n_splits = 2

    #Optimizer and Scheduler args
    loss = 'nn.BCEWithLogitsLoss'
    lr = 3e-4
    warmup_epochs = 5
    weight_decay = 3e-6
    eta_min = 0.000001
    n_accumulate = 1
    T_0 = 25
    T_max = 2000

    #Callback args
    #Minimum number amount of improvement to not trigger patience
    min_delta = 0.0
    #Number of epochs in a row to wait for improvement
    patience = 30

#Dataloader Args
loaderargs = {'num_workers' : args.num_workers, 'pin_memory': False, 'drop_last': False}
device = torch.device("cuda:0")

seed_everything(args.seed)

from pickle import load

target_scaler = load(open('scaler.pkl', 'rb'))
transform = T.Resize(size = (224,224))

def image_transform(image):
    image = np.swapaxes(image, 0, 2)
    image = torch.from_numpy(image)
    image = torch.div(image, 255)
    image = transform(image)
    return image

def get_default_transforms():
    transform = {
        "train": T.Compose(
            [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean = (0.485, 0.456, 0.406), 
                            std = (0.229, 0.224, 0.225))
                
            ]
        ),
        "val": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean = (0.485, 0.456, 0.406), 
                            std = (0.229, 0.224, 0.225))
            ]
        ),
    }
    return transform

#Model
class CNNLSTM(LightningModule):
    def __init__(self):
        super().__init__()
        self.args = args
        self.scaler = target_scaler
        self._criterion = eval(self.args.loss)()
        self.transform = get_default_transforms()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=.2)
                
        #Tracking
        self.trainr2 = R2Score()
        self.valr2 = R2Score()
        
        #Image Models
        self.model1 = timm.create_model(args.model_name1, 
                                       pretrained=args.pretrained, 
                                       num_classes=0,
                                       in_chans = 3,
                                       #global_pool=''
                                       )
        self.model2 = timm.create_model(args.model_name2, 
                                       pretrained=args.pretrained, 
                                       num_classes=0,
                                       in_chans = 3,
                                       #global_pool=''
                                       )
        self.model3 = timm.create_model(args.model_name2, 
                                       pretrained=args.pretrained, 
                                       num_classes=0,
                                       in_chans = 3,
                                       #global_pool=''
                                       )
        self.model4 = timm.create_model(args.model_name2, 
                                       pretrained=args.pretrained, 
                                       num_classes=0,
                                       in_chans = 3,
                                       #global_pool=''
                                       )
        #LSTM
        self.lstm = nn.LSTM(input_size = 5,
                            hidden_size = args.lstm_hidden,
                            num_layers = args.lstm_layers,
                            batch_first=True,dropout=.1)
        #Possible multiple LSTM layers?
        #self.lstm = nn.LSTM(input_size = args.lstm_hidden,
        #                    hidden_size = self.hidden_size,
        ##                    num_layers = args.lstm_layers,
        #                    batch_first=True,dropout=.1)
        
        #Linear regression layer
        self.linear1 = nn.Linear(7046,1024)
        self.linear2 = nn.Linear(1024,256)
        self.linear3 = nn.Linear(256,args.num_classes)
    
        
    def forward(self,features1,features2,features3,features4,meta,ts):
        
        
        
        #Image Convolution
        #Image Models
        features1 = self.model1(features1)                 
        features1 = self.relu(features1)
        features1 = self.dropout(features1)
        
        features2 = self.model2(features2)                 
        features2 = self.relu(features2)
        features2 = self.dropout(features2)
        
        features3 = self.model3(features3)                 
        features3 = self.relu(features3)
        features3 = self.dropout(features3)
        
        features4 = self.model4(features4)                 
        features4 = self.relu(features4)
        features4 = self.dropout(features4)
        

        #LSTM
        batch_size, seq_len, feature_len = ts.size()
        # Initialize hidden state with zeros
        
        h_0 = torch.zeros(2, batch_size, args.lstm_hidden,requires_grad=True).cuda()
        c_0 = torch.zeros(2, batch_size, args.lstm_hidden ,requires_grad=True).cuda()
        
        f_ts, (final_hidden,final_cell) = self.lstm(ts, (h_0,c_0))
        f_ts = f_ts.contiguous().view(batch_size,-1)
        
        #*************************************************************
        #Concatenate meta and image features
        features = torch.cat([features1,features2,features3,features4,f_ts,meta],dim=1)
        #*************************************************************
        
        #Linear
        features = self.linear1(features)
        features = self.relu(features)
        features = self.dropout(features)
        
        features = self.linear2(features)
        features = self.relu(features)
        features = self.dropout(features)
        
        output = self.linear3(features)
        return output
    
###I DIDN"T MIX UP TS data
    def __share_step(self, batch, mode):
        copernicus_img, sentinel1_img, sentinel2a_img, sentinel2b_img, labels, meta,ts = batch
        labels = labels.float()
        meta = meta.float()
        ts = ts.float()
        copernicus_img = self.transform[mode](copernicus_img)
        sentinel1_img = self.transform[mode](sentinel1_img)
        sentinel2a_img = self.transform[mode](sentinel2a_img)
        sentinel2b_img = self.transform[mode](sentinel2b_img)

        

        logits = self.forward(copernicus_img,sentinel1_img,sentinel2a_img,sentinel2b_img, meta,ts).squeeze(1)
        loss = self._criterion(logits, labels)

        pred = torch.from_numpy(self.scaler \
            .inverse_transform(np.array(logits.sigmoid().detach().cpu()) \
            .reshape(-1, 1)))
        labels = torch.from_numpy(self.scaler \
            .inverse_transform(np.array(labels.detach().cpu()) \
            .reshape(-1, 1)))
        
        '''
        #This is random noise
        elif rand_index > 0.8 and mode == 'train':
            images = images + (torch.randn(images.size(0),3,args.imagesize,args.imagesize, 
                                           dtype = torch.float, device = device)*10)/100
            logits = self.forward(images, meta).squeeze(1)
            loss = self._criterion(logits, labels)
        '''

        return loss, pred, labels

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, 'train')
        self.trainr2(pred.cuda(),labels.cuda())
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {'loss': loss, 'pred': pred, 'labels': labels}
    
    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, 'val')
        self.valr2(pred.cuda(),labels.cuda())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'pred': pred, 'labels': labels}

    def training_epoch_end(self, outputs):
        self.log('train_r2_epoch',self.trainr2)
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.log('val_r2_epoch',self.valr2)
        self.__share_epoch_end(outputs, 'val')

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['pred'], out['labels']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f'{mode}_RMSE', metrics)    


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=args.lr, weight_decay = args.weight_decay)
        
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": CosineAnnealingLR(optimizer, T_max = args.T_max, eta_min= args.eta_min),
            "interval": "step",
            "monitor": "train_loss",
            "frequency": 1}
            }

print("Creating Model...")
model = CNNLSTM().load_from_checkpoint("best_weights.ckpt").to(device)
print("Model Created")
##########Prediction module####################

predictions = []

x = 0

totalstart = time.time()

transforms = get_default_transforms()

print("Predicting")

for i in range(len(df)):
    if np.isnan(df.SWE[i]):
        predictions.append(np.nan)
        x+= 1
    else:
        start = time.time()
        MOD10A1 = data_wrangling.pull_MODIS_list(df.geometry[i], date, 'MOD10A1')
        MYD10A1 = data_wrangling.pull_MODIS_list(df.geometry[i], date, 'MYD10A1')
        modis = MOD10A1[2::] + MYD10A1[2::]
        modis = torch.FloatTensor(modis).unsqueeze(0).to(device)
        
        copernicus = data_wrangling.get_copernicus(df.geometry[i])
        copernicus = image_transform(copernicus)
        copernicus = transforms["val"](copernicus).unsqueeze(0).to(device)
        
        sen1 = data_wrangling.pull_Sentinel1(df.geometry[i], date)
        sen1 = image_transform(sen1)
        sen1 = transforms["val"](sen1).unsqueeze(0).to(device)
        
        
        sen2a = data_wrangling.pull_Sentinel2a(df.geometry[i], date)
        sen2a = image_transform(sen2a)
        sen2a = transforms["val"](sen2a).unsqueeze(0).to(device) 
        
        
        sen2b = data_wrangling.pull_Sentinel2b(df.geometry[i], date)
        sen2b = image_transform(sen2b)
        sen2b = transforms["val"](sen2b).unsqueeze(0).to(device)  
        
        
        weather = data_wrangling.pull_GRIDMET(df.geometry[i], date, num_days_back = 10)
        weather['date'] = pd.to_datetime(weather['date'])
        sequence = weather.fillna(-1)
        
        daterange = pd.date_range(end = date, periods = 10).tolist()

        if list(sequence.shape) != [10,7]:
            missing_dates = [[-1,i,-1,-1,-1,-1,-1] for i in daterange if i not in list(sequence.date)]
            missing_dates = pd.DataFrame(data = missing_dates,columns=['geometry','date','precip','wind_dir','temp_min','temp_max','wind_vel'])
            sequence = pd.concat([sequence,missing_dates],axis=0,ignore_index=True).sort_values('date')
        #Drop non-tabular data columns
        sequence.drop(['geometry','date'],axis=1,inplace=True)
            

        ts = torch.tensor(sequence.values.tolist(),dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(copernicus, sen1, sen2a, sen2b, modis, ts).squeeze(1)
        
        pred =target_scaler  \
            .inverse_transform(np.array(output.sigmoid().detach().cpu()) \
            .reshape(-1,1))
        
        print(pred[0])
        
        predictions.extend(pred[0])
        x+=1
        
        print(f'{round(x/len(df)*100,3)}% -- {x} out of {len(df)} km complete')
        print(f"Current time per km: {round(time.time()-start, 3)} seconds")

df.SWE = predictions

print(f"Prediction complete! Total time was {round(time.time()-totalstart, 3)} seconds")
print("Stitching image....")

im_array = data_wrangling.stitch_aso(tif_path, df, date = str(date))

def filter_nan_gaussian_conserving(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss



sigma = 200
plt.clf()
plt.close()
blurred =filter_nan_gaussian_conserving(im_array, sigma)

ax = sns.heatmap(blurred, vmin = 0, vmax = 100, cmap = "mako_r", yticklabels=False, xticklabels=False)
plt.savefig(f"{tif_path[0:-7]}{date}_smoothed_prediction.png")
np.savetxt(f"{tif_path[0:-7]}_{date}_smoothed_prediction.csv", blurred, delimiter=",")

print('Your data is now in the snowcast_prediction/ReferenceImages directory')







