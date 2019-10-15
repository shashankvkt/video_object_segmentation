import torch
import torch.utils.data
from torch import nn, optim
import torchvision.models as models
from torch.autograd import Variable
from torch.nn import functional as F

'''
Consists of the Initializer, Encoder and Decoder module 

'''

class Initializer(nn.Module):
	def __init__(self):
		super(Initializer, self).__init__()
		self.new_layer = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3)
		

		self.pretrained_model = models.vgg16(pretrained=True)
		self.model = nn.Sequential(*list(self.pretrained_model.features.children())[2:31])
		self.interp = nn.functional.interpolate

		self.c0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
		

		self.h0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
		
		self.relu = nn.ReLU()

	def forward(self, inputs):
		x = self.relu(self.new_layer(inputs))
		x = self.model(x)
		c0 = self.relu(self.c0(x))
		h0 = self.relu(self.h0(x))
		c0 = self.interp(c0, size=(8,14), mode='bilinear', align_corners=False)
		h0 = self.interp(h0, size=(8,14), mode='bilinear', align_corners=False)
		return c0,h0


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.pretrained_model = models.vgg16(pretrained=True)
		self.model = nn.Sequential(*list(self.pretrained_model.features.children())[0:31])
		self.avgPool = nn.AvgPool2d(kernel_size = 8, stride = 0, padding = 0, ceil_mode=False, count_include_pad=True)
		self.relu = nn.ReLU()
		
	def forward(self, inputs):
		x = self.model(inputs)
		x = self.relu((x))
		return x


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.interp = nn.functional.interpolate

		self.deconv1 =  nn.ConvTranspose2d(512,256, 5,2)
		
		self.relu=nn.ReLU()

		self.deconv2 = nn.ConvTranspose2d(256,128, 5,2)
		

		self.deconv3 = nn.ConvTranspose2d(128,64, 5,2)
		

		self.deconv4 = nn.ConvTranspose2d(64,64, 5,2)
		

		self.deconv5 =  nn.ConvTranspose2d(64,64, 5,2)
		
		self.conv = nn.Conv2d(64, 1, 5,2,1)
		
		self.sigmoid = nn.Sigmoid()

	def forward(self, inputs):
		
		x = self.deconv1(inputs)
		x = self.relu(x)
		
		x = self.deconv2(x)
		x = self.relu(x)
		
		x = self.deconv3(x)
		x = self.relu(x)
		x = self.deconv4(x)
		x = self.relu(x)
		x = self.deconv5(x)
		x = self.relu(x)
		
		x = self.conv(x)
		x = self.interp(x, size=(256,448), mode='bilinear', align_corners=False)
		x = self.sigmoid(x)
		
		return x


