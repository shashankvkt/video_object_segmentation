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
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import glob
from torchvision.utils import save_image
import configuration as cfg
from utils.dataloader5fps import dataLoader_5fps
from utils.initializer import Initializer, Encoder, Decoder
from utils.convlstm import *
from utils.ensemble import *

json_file = open(cfg.validation_json)
json_str = json_file.read()
json_data = json.loads(json_str)
validFolders = os.listdir(cfg.JPEGValidation)


transform_rgb = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop((256,448)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

transforms_seg = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop((256,448)),
		transforms.ToTensor() ])




initializer = Initializer()
encoder = Encoder()
convlstm = ConvLSTMCell(input_size=512,
				 hidden_size=512)
decoder = Decoder()


model = MyEnsemble(initializer,encoder,convlstm,decoder)

model.load_state_dict(torch.load(cfg.modelPth))

model.cuda()


saveDir = '/home/shashank/shashank/AdvCV/YoutubeVOS_submission/'

def test():
	model.eval()
	for i in range(len(validFolders)):
		selectFolder = validFolders[i]
		segPixel = (list(json_data['videos'][selectFolder]['objects'].keys()))
		segpth = cfg.AnnValidation + selectFolder + '/'
		rgbpth = cfg.JPEGValidation + selectFolder + '/'
		for j in range(len(segPixel)):
			selectSegPixel = segPixel[j]
			frames = json_data['videos'][selectFolder]['objects'][str(selectSegPixel)]['frames']

			initialseg = np.array(Image.open(segpth + frames[0] + '.png'))
			intialmask = np.zeros((initialseg.shape))
			indices = np.where(initialseg == int(selectSegPixel))
			intialmask[indices[0],indices[1]] = 255
			intialmask = transforms_seg(Image.fromarray(np.uint8(intialmask)))
			intialmask = intialmask.unsqueeze(0).cuda()

			initRGB = transform_rgb(Image.open(rgbpth + frames[0] + '.jpg'))
			initRGB = initRGB.unsqueeze(0).cuda()

			directory = saveDir + selectFolder + '/' + segPixel[j] + '/' 
			print(directory)



			if not os.path.exists(directory):
				os.makedirs(directory)

			for k in range(len(frames)):
				rgbImg = transform_rgb(Image.open(rgbpth + frames[k] + '.jpg'))
				rgbImg = rgbImg.unsqueeze(0).type(torch.FloatTensor).cuda()

				output = model(initRGB, intialmask, rgbImg)
				output = nn.functional.interpolate(output, size=(720,1280), mode='bilinear', align_corners=False)

				# temp = 1.0*(output > 0.4)
				output = output.squeeze()
				output = output.data.cpu().numpy()
				output = 255*output

				path = directory +  frames[k] + '.png'
				cv2.imwrite(path,output)
				


if __name__ == '__main__':
	test()



