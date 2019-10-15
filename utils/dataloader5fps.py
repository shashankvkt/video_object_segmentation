import os
import sys
import numpy as np
import cv2
import json
import random
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import glob

'''
Taking every 5th frame for training
'''

class dataLoader_5fps(Dataset):
	def __init__(self, JPEGPath, AnnPath, json_path, transform_rgb=None, transform_seg=None):
		self.JPEGPath = JPEGPath
		self.AnnPath = AnnPath
		self.json_path = json_path
		self.transform_rgb = transform_rgb
		self.transform_seg = transform_seg
		self.folders = os.listdir(self.JPEGPath)
		json_file = open(self.json_path)
		json_str = json_file.read()
		self.json_data = json.loads(json_str)

		
	def __getitem__(self,index):
		selectFolder = self.folders[index]
		segPixel = (list(self.json_data['videos'][selectFolder]['objects'].keys()))
		selectSegPixel = random.choice(segPixel)
		frames = self.json_data['videos'][selectFolder]['objects'][str(selectSegPixel)]['frames']
		try:
			start = random.randint(0,len(frames)-5)
		except:
			start = 0

		end = start + 5
		if(end > len(frames)):
			end = len(frames)
		count = 0
		rgbFrames = np.zeros((5,3,256,448))
		maskedSegFrame = np.zeros((5,1,256,448))
		for i in range(start, end):
			
			segpth = self.AnnPath + selectFolder + '/' + frames[i] + '.png'
			rgbpth = self.JPEGPath + selectFolder + '/' + frames[i] + '.jpg'
			
			img =  np.array(Image.open(segpth))
			mask = np.zeros((img.shape))
			indices = np.where(img == int(selectSegPixel))
			mask[indices[0],indices[1]] = 255
			mask = Image.fromarray(np.uint8(mask))
			if(self.transform_seg):
			 	mask = self.transform_seg(mask)
			maskedSegFrame[count,:,:,:] = (mask)

			img_rgb = Image.open(rgbpth)
			if(self.transform_rgb):
				img_rgb = self.transform_rgb(img_rgb)
			rgbFrames[count,:,:,:] = (img_rgb)
			count += 1

		count=0
		maskedSegFrame =  torch.from_numpy(maskedSegFrame)        	
		rgbFrames =  torch.from_numpy(rgbFrames)

		initialrgb = Image.open(self.JPEGPath + selectFolder + '/' + frames[0] + '.jpg')
		if(self.transform_rgb):
				initialrgb = self.transform_rgb(initialrgb)

		initialseg = np.array(Image.open(self.AnnPath + selectFolder + '/' + frames[0] + '.png'))
		intialmask = np.zeros((initialseg.shape))
		indices = np.where(initialseg == int(selectSegPixel))
		intialmask[indices[0],indices[1]] = 255
		intialmask = Image.fromarray(np.uint8(intialmask))
		if(self.transform_seg):
		 	intialmask = self.transform_seg(intialmask)
		

		return intialmask, initialrgb, maskedSegFrame, rgbFrames
	
	def __len__(self): 
		return len(self.folders)

