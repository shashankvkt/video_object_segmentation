
import torch
import torch.utils.data
from torch import nn, optim
import torchvision.models as models
from torch.autograd import Variable
from torch.nn import functional as F

'''

Encoder - ConvLSTM - Decoder module

'''


class MyEnsemble(nn.Module):
    def __init__(self, initializer, encoder,convlstm,decoder):
        super(MyEnsemble, self).__init__()
        self.initializer = initializer
        self.encoder = encoder
        self.convlstm = convlstm
        self.decoder = decoder
        
    def forward(self, initRGB, initMask, RGBData):
        predictedMask = []
        c0,h0 = self.initializer(torch.cat((initRGB,initMask),1))
        for i in range(5):
            rgbFrame = RGBData[:,i,:,:,:]
            x_tilda = self.encoder(rgbFrame)
            c_next,h_next = self.convlstm(x_tilda, h0, c0)
            output = self.decoder(h_next)
            c0 = c_next
            h0 = h_next
            predictedMask.append(output)
        predictedMask = torch.stack(predictedMask).type(torch.FloatTensor).cuda()
        predictedMask = predictedMask.transpose(1,0)

        return predictedMask



