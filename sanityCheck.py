
import os
import sys
import numpy as np
import cv2
import json
import random
import torch
import torch.utils.data
import torchvision.models as models
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import configuration as cfg


json_file = open(cfg.validation_json)
json_str = json_file.read()
json_data = json.loads(json_str)
validFolders = os.listdir(cfg.indivAnnotation)



for i in range(len(validFolders)):
		selectFolder = validFolders[i]
		segPixel = (list(json_data['videos'][selectFolder]['objects'].keys()))
		
		rgbpth = cfg.indivAnnotation_check + 'Annotations/' + selectFolder + '/'
		for j in range(len(segPixel)):
			selectSegPixel = segPixel[j]
			frames = json_data['videos'][selectFolder]['objects'][str(selectSegPixel)]['frames']

			
			for k in range(len(frames)):
				print(rgbpth + frames[k] + '.png',i)
				rgbImg = (Image.open(rgbpth + frames[k] + '.png'))
				exit()



