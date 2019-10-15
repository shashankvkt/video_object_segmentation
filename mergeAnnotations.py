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
videoFolders = os.listdir(cfg.indivAnnotation)
saveDir = '/home/shashank/shashank/AdvCV/YoutubeVOS_merged_thrsh04/Annotations/'
readpth = '/home/shashank/shashank/AdvCV/YoutubeVOS_submission/'

test_image = Image.open('/home/shashank/shashank/datasets/YoutubeVos/train-004/train/Annotations/a36bdc4cab/00000.png')



for i in range(len(videoFolders)):#
		#print(i)
		selectFolder = videoFolders[i] #'03faedf2a3'
		print(selectFolder,i)
		#exit(0)
		segPixel = (list(json_data['videos'][selectFolder]['objects'].keys()))
		totalFrames = []
		
		savepth = saveDir + selectFolder + '/'
		

		if not os.path.exists(savepth):
				os.makedirs(savepth)

		for j in range(len(segPixel)):
			selectSegPixel = segPixel[j]
			frames = json_data['videos'][selectFolder]['objects'][str(selectSegPixel)]['frames']
			totalFrames.append(frames)

		if(len(totalFrames) == 2):
			
			frame_a = set(totalFrames[0])
			frame_b = set(totalFrames[1])
			common = sorted(frame_a & frame_b)

			uncommon = sorted(frame_a - frame_b)
			
			read1 = readpth + selectFolder + '/' + segPixel[0] + '/'
			read2 = readpth + selectFolder + '/' + segPixel[1] + '/'
			

			#------------------- COMMON LEN = 2 ------------------------------------------------------
			for k in range(len(common)):
				img1 = cv2.imread(read1 + common[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img2 = cv2.imread(read2 + common[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))
				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img2,background))
				pixel = [(segPixel[0]),(segPixel[1]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img2[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

				
				
			#--------------------- UNCOMMON LEN = 2 ------------------------------------------------------
			for k in range(len(uncommon)):
				if(uncommon[k] in totalFrames[0]):
					read = readpth + selectFolder + '/' + segPixel[0] + '/'
					img = cv2.imread(read + uncommon[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[0]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)
					
				else:
					read = readpth + selectFolder + '/' + segPixel[1] + '/'
					img = cv2.imread(read + uncommon[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[1]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)

		
				
		elif(len(totalFrames) == 1):
			read = readpth + selectFolder + '/' + segPixel[0] + '/'
			frames = totalFrames[0]
			for k in range(len(frames)):
				img = cv2.imread(read + frames[k] + '.png')
				img = img[:,:,1]
				img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

				
				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img,background))
				pixel = [(segPixel[0]),0]

				
				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + frames[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)
				

		elif(len(totalFrames) == 3):
			frame_a = set(totalFrames[0])
			frame_b = set(totalFrames[1])
			frame_c = set(totalFrames[2])

			common = (frame_a & frame_b & frame_c)
			common_12 = sorted((frame_a & frame_b) - common)
			common_23 = sorted((frame_b & frame_c) - common)
			common_13 = sorted((frame_a & frame_c) - common)
			#common = sorted(common)
		

			uncommon_three = sorted((frame_a | frame_b | frame_c) - common)
			common = sorted(common)
			
		
			#uncommon_12 = (frame_a - frame_b)
			#uncommon_23 = (frame_b - frame_c)
			#uncommon_13 = (frame_a - frame_c)


			read1 = readpth + selectFolder + '/' + segPixel[0] + '/'
			read2 = readpth + selectFolder + '/' + segPixel[1] + '/'
			read3 = readpth + selectFolder + '/' + segPixel[2] + '/'

			for k in range(len(common)):
				img1 = cv2.imread(read1 + common[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img2 = cv2.imread(read2 + common[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))

				img3 = cv2.imread(read3 + common[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img2,img3,background))
				pixel = [(segPixel[0]),(segPixel[1]),(segPixel[2]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img2[row,col],img3[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			

			for k in range(len(common_12)):
				img1 = cv2.imread(read1 + common_12[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img2 = cv2.imread(read2 + common_12[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))

				

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img2,background))
				pixel = [(segPixel[0]),(segPixel[1]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img2[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_12[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)


			for k in range(len(common_23)):
				
				img2 = cv2.imread(read2 + common_23[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))

				img3 = cv2.imread(read3 + common_23[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img2,img3,background))
				pixel = [(segPixel[1]),(segPixel[2]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img2[row,col],img3[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_23[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_13)):
				img1 = cv2.imread(read1 + common_13[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))


				img3 = cv2.imread(read3 + common_13[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img3,background))
				pixel = [(segPixel[0]),(segPixel[2]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img3[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_13[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(uncommon_three)):
				
				if(uncommon_three[k] in totalFrames[0]):
					read = readpth + selectFolder + '/' + segPixel[0] + '/'
					img = cv2.imread(read + uncommon_three[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[0]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_three[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)
					
				elif(uncommon_three[k] in totalFrames[1]):
					read = readpth + selectFolder + '/' + segPixel[1] + '/'
					img = cv2.imread(read + uncommon_three[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[1]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_three[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)
				else:
					read = readpth + selectFolder + '/' + segPixel[2] + '/'
					img = cv2.imread(read + uncommon_three[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[2]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_three[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)


		elif(len(totalFrames) == 4):
			frame_a = set(totalFrames[0])
			frame_b = set(totalFrames[1])
			frame_c = set(totalFrames[2])
			frame_d = set(totalFrames[3])

			common = (frame_a & frame_b & frame_c & frame_d)
			common_12 = sorted((frame_a & frame_b) - common)
			common_13 = sorted((frame_a & frame_c) - common)
			common_14 = sorted((frame_a & frame_d) - common)
			common_23 = sorted((frame_b & frame_c) - common)
			common_24 = sorted((frame_b & frame_d) - common)
			common_34 = sorted((frame_c & frame_d) - common)
			#common = sorted(common)
			

			uncommon_four = sorted((frame_a | frame_b | frame_c | frame_d) - common)
			common = sorted(common)
			#uncommon_12 = (frame_a - frame_b)
			#uncommon_23 = (frame_b - frame_c)
			#uncommon_13 = (frame_a - frame_c)


			read1 = readpth + selectFolder + '/' + segPixel[0] + '/'
			read2 = readpth + selectFolder + '/' + segPixel[1] + '/'
			read3 = readpth + selectFolder + '/' + segPixel[2] + '/'
			read4 = readpth + selectFolder + '/' + segPixel[3] + '/'

			for k in range(len(common)):
				img1 = cv2.imread(read1 + common[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img2 = cv2.imread(read2 + common[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))

				img3 = cv2.imread(read3 + common[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				img4 = cv2.imread(read4 + common[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img2,img3,img4,background))
				pixel = [(segPixel[0]),(segPixel[1]),(segPixel[2]),(segPixel[3]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img2[row,col],img3[row,col],img4[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			

			for k in range(len(common_12)):
				img1 = cv2.imread(read1 + common_12[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img2 = cv2.imread(read2 + common_12[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))

				

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img2,background))
				pixel = [(segPixel[0]),(segPixel[1]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img2[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_12[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)


			for k in range(len(common_13)):
				
				img1 = cv2.imread(read1 + common_13[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img3 = cv2.imread(read3 + common_13[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img3,background))
				pixel = [(segPixel[0]),(segPixel[2]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img3[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_13[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_14)):
				img1 = cv2.imread(read1 + common_14[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))


				img4 = cv2.imread(read4 + common_14[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img4,background))
				pixel = [(segPixel[0]),(segPixel[3]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img4[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_14[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_23)):
				img2 = cv2.imread(read2 + common_23[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))


				img3 = cv2.imread(read3 + common_23[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img2,img3,background))
				pixel = [(segPixel[1]),(segPixel[2]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img2[row,col],img3[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_23[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_24)):
				img2 = cv2.imread(read2 + common_24[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))


				img4 = cv2.imread(read4 + common_24[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img2,img4,background))
				pixel = [(segPixel[1]),(segPixel[3]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img2[row,col],img4[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_24[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_34)):
				img3 = cv2.imread(read3 + common_34[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))


				img4 = cv2.imread(read4 + common_34[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img3,img4,background))
				pixel = [(segPixel[2]),(segPixel[3]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img3[row,col],img4[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_34[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)



			for k in range(len(uncommon_four)):
				
				if(uncommon_four[k] in totalFrames[0]):
					read = readpth + selectFolder + '/' + segPixel[0] + '/'
					img = cv2.imread(read + uncommon_four[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[0]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_four[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)
					
				elif(uncommon_four[k] in totalFrames[1]):
					read = readpth + selectFolder + '/' + segPixel[1] + '/'
					img = cv2.imread(read + uncommon_four[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[1]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_four[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)

				elif(uncommon_four[k] in totalFrames[2]):
					read = readpth + selectFolder + '/' + segPixel[2] + '/'
					img = cv2.imread(read + uncommon_four[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[1]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_four[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)

				else:
					read = readpth + selectFolder + '/' + segPixel[3] + '/'
					img = cv2.imread(read + uncommon_four[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[2]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_four[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)

#------------------------- LEN = 5 --------------------------------------------------
#--------------------------------------------------------------------------------------

		elif(len(totalFrames) == 5):
			frame_a = set(totalFrames[0])
			frame_b = set(totalFrames[1])
			frame_c = set(totalFrames[2])
			frame_d = set(totalFrames[3])
			frame_e = set(totalFrames[4])

			common = (frame_a & frame_b & frame_c & frame_d & frame_e)
			common_12 = sorted((frame_a & frame_b) - common)
			common_13 = sorted((frame_a & frame_c) - common)
			common_14 = sorted((frame_a & frame_d) - common)
			common_15 = sorted((frame_a & frame_e) - common)
			common_23 = sorted((frame_b & frame_c) - common)
			common_24 = sorted((frame_b & frame_d) - common)
			common_25 = sorted((frame_b & frame_e) - common)
			common_34 = sorted((frame_c & frame_d) - common)
			common_35 = sorted((frame_c & frame_e) - common)
			common_45 = sorted((frame_d & frame_e) - common)
			#common = sorted(common)
			

			uncommon_five = sorted((frame_a | frame_b | frame_c | frame_d | frame_e) - common)
			common = sorted(common)
			


			read1 = readpth + selectFolder + '/' + segPixel[0] + '/'
			read2 = readpth + selectFolder + '/' + segPixel[1] + '/'
			read3 = readpth + selectFolder + '/' + segPixel[2] + '/'
			read4 = readpth + selectFolder + '/' + segPixel[3] + '/'
			read5 = readpth + selectFolder + '/' + segPixel[4] + '/'

			for k in range(len(common)):
				img1 = cv2.imread(read1 + common[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img2 = cv2.imread(read2 + common[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))

				img3 = cv2.imread(read3 + common[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				img4 = cv2.imread(read4 + common[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				img5 = cv2.imread(read5 + common[k] + '.png')
				img5 = img5[:,:,1]
				img5 = (img5 - np.amin(img5)) / (np.amax(img5) - np.amin(img5))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img2,img3,img4,img5,background))
				pixel = [(segPixel[0]),(segPixel[1]),(segPixel[2]),(segPixel[3]),(segPixel[4]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img2[row,col],img3[row,col],img4[row,col],img5[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			

			for k in range(len(common_12)):
				img1 = cv2.imread(read1 + common_12[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img2 = cv2.imread(read2 + common_12[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))

				

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img2,background))
				pixel = [(segPixel[0]),(segPixel[1]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img2[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_12[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)


			for k in range(len(common_13)):
				
				img1 = cv2.imread(read1 + common_13[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))

				img3 = cv2.imread(read3 + common_13[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img3,background))
				pixel = [(segPixel[0]),(segPixel[2]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img3[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_13[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_14)):
				img1 = cv2.imread(read1 + common_14[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))


				img4 = cv2.imread(read4 + common_14[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img4,background))
				pixel = [(segPixel[0]),(segPixel[3]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img4[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_14[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_15)):
				img1 = cv2.imread(read1 + common_15[k] + '.png')
				img1 = img1[:,:,1]
				img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))


				img5 = cv2.imread(read5 + common_15[k] + '.png')
				img5 = img5[:,:,1]
				img5 = (img5 - np.amin(img5)) / (np.amax(img5) - np.amin(img5))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img1,img4,background))
				pixel = [(segPixel[0]),(segPixel[4]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img1[row,col],img5[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_15[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_23)):
				img2 = cv2.imread(read2 + common_23[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))


				img3 = cv2.imread(read3 + common_23[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img2,img3,background))
				pixel = [(segPixel[1]),(segPixel[2]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img2[row,col],img3[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_23[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_24)):
				img2 = cv2.imread(read2 + common_24[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))


				img4 = cv2.imread(read4 + common_24[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img2,img4,background))
				pixel = [(segPixel[1]),(segPixel[3]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img2[row,col],img4[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_24[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_25)):
				img2 = cv2.imread(read2 + common_25[k] + '.png')
				img2 = img2[:,:,1]
				img2 = (img2 - np.amin(img2)) / (np.amax(img2) - np.amin(img2))


				img5 = cv2.imread(read5 + common_25[k] + '.png')
				img5 = img5[:,:,1]
				img5 = (img5 - np.amin(img5)) / (np.amax(img5) - np.amin(img5))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img2,img5,background))
				pixel = [(segPixel[1]),(segPixel[4]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img2[row,col],img5[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_25[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_34)):
				img3 = cv2.imread(read3 + common_34[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))


				img4 = cv2.imread(read4 + common_34[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img3,img4,background))
				pixel = [(segPixel[2]),(segPixel[3]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img3[row,col],img4[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_34[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_35)):
				img3 = cv2.imread(read3 + common_35[k] + '.png')
				img3 = img3[:,:,1]
				img3 = (img3 - np.amin(img3)) / (np.amax(img3) - np.amin(img3))

				img5 = cv2.imread(read5 + common_35[k] + '.png')
				img5 = img5[:,:,1]
				img5 = (img5 - np.amin(img5)) / (np.amax(img5) - np.amin(img5))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img3,img5,background))
				pixel = [(segPixel[2]),(segPixel[4]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img3[row,col],img5[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_35[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)

			for k in range(len(common_45)):
				img4 = cv2.imread(read4 + common_45[k] + '.png')
				img4 = img4[:,:,1]
				img4 = (img4 - np.amin(img4)) / (np.amax(img4) - np.amin(img4))

				img5 = cv2.imread(read5 + common_45[k] + '.png')
				img5 = img5[:,:,1]
				img5 = (img5 - np.amin(img5)) / (np.amax(img5) - np.amin(img5))

				background = np.zeros((720,1280)) + cfg.thresh
				temp = np.dstack((img4,img5,background))
				pixel = [(segPixel[3]),(segPixel[4]),0]

				finalImage = Image.new('P', (720,1280))
				finalImagePixel = finalImage.load()

				for row in range(720):
					for col in range(1280):
						values = [img3[row,col],img5[row,col],background[row,col]]
						maxVal = max(values)
						indx = values.index(maxVal)
			

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
						

				finalImage = finalImage.transpose(Image.TRANSPOSE)
				finalImage = np.array(finalImage, dtype=np.int8)
				finalImage = Image.fromarray(finalImage, mode='P')
				finalImage.putpalette(test_image.getpalette())

				saveImg = savepth + common_45[k] + '.png'
				print(saveImg)
				finalImage.save(saveImg)




			for k in range(len(uncommon_five)):
				
				if(uncommon_five[k] in totalFrames[0]):
					read = readpth + selectFolder + '/' + segPixel[0] + '/'
					img = cv2.imread(read + uncommon_five[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[0]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_five[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)
					
				elif(uncommon_five[k] in totalFrames[1]):
					read = readpth + selectFolder + '/' + segPixel[1] + '/'
					img = cv2.imread(read + uncommon_five[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[1]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_five[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)

				elif(uncommon_five[k] in totalFrames[2]):
					read = readpth + selectFolder + '/' + segPixel[2] + '/'
					img = cv2.imread(read + uncommon_five[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[1]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_five[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)

				elif(uncommon_five[k] in totalFrames[3]):
					read = readpth + selectFolder + '/' + segPixel[3] + '/'
					img = cv2.imread(read + uncommon_five[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[1]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)


						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_five[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)

				else:
					read = readpth + selectFolder + '/' + segPixel[4] + '/'
					img = cv2.imread(read + uncommon_five[k] + '.png')
					img = img[:,:,1]
					img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

					
					background = np.zeros((720,1280)) + cfg.thresh
					temp = np.dstack((img,background))
					pixel = [(segPixel[2]),0]

					finalImage = Image.new('P', (720,1280))
					finalImagePixel = finalImage.load()

					for row in range(720):
						for col in range(1280):
							values = [img[row,col],background[row,col]]
							maxVal = max(values)
							indx = values.index(maxVal)

						if(int(pixel[indx]) == 0):
							Val = 0#(0, 0, 0)
						elif(int(pixel[indx]) == 1):
							Val = 1#(236, 95, 103)
						elif(int(pixel[indx]) == 2):
							Val = 2#(249, 145, 87)
						elif(int(pixel[indx]) == 3):
							Val = 3#(250, 200, 99)
						elif(int(pixel[indx]) == 4):
							Val = 4#(153, 199, 148)
						elif(int(pixel[indx]) == 5):
							Val = 5#(98, 179, 178)
						elif(int(pixel[indx]) == 6):
							Val = 6#(102, 153, 204)

						finalImagePixel[row,col] = Val
							

					finalImage = finalImage.transpose(Image.TRANSPOSE)
					finalImage = np.array(finalImage, dtype=np.int8)
					finalImage = Image.fromarray(finalImage, mode='P')
					finalImage.putpalette(test_image.getpalette())

					saveImg = savepth + uncommon_five[k] + '.png'
					print(saveImg)
					finalImage.save(saveImg)








		