import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class GoNet(nn.Module):
    """ Neural Network class
        Two stream model:
        ________
       |        | conv layers              Untrained Fully
       |Previous|------------------>|      Connected Layers
       | frame  |                   |    ___     ___     ___
       |________|                   |   |   |   |   |   |   |   fc4
                   Pretrained       |   |   |   |   |   |   |    * (left)
                   CaffeNet         |-->|fc1|-->|fc2|-->|fc3|--> * (top)
                   Convolution      |   |   |   |   |   |   |    * (right)
                   layers           |   |___|   |___|   |___|    * (bottom)
        ________                    |   (4096)  (4096)  (4096)  (4)
       |        |                   |
       | Current|------------------>|
       | frame  |
       |________|

    """
    def __init__(self):
        super(GoNet, self).__init__()
        #caffenet = models.alexnet(pretrained=True)
        caffenet = models.vgg16(pretrained=True)
        self.convnet = nn.Sequential(*list(caffenet.children())[:-1])
        for param in self.convnet.parameters():
            param.requires_grad = False
        """
        caffenetcorr = models.alexnet(pretrained=False)
        self.convnetcorr = nn.Sequential(*list(caffenetcorr.children())[:-1])
        for param in self.convnetcorr.parameters():
            param.requires_grad = True 
        """
        self.classifier = nn.Sequential(
                #nn.Linear(256*6*6*2, 4096), #
                nn.Linear(512*7*7*2, 4096), #
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4),
                )
        self.weight_init()

    def weight_init(self):
        for m in self.classifier.modules():
            # fully connected layers are weight initialized with
            # mean=0 and std=0.005 (in tracker.prototxt) and
            # biases are set to 1
            # tracker.prototxt link: https://goo.gl/iHGKT5
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)

    def forward(self, x, y):
        
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cat = torch.zeros(x.shape).to(device)
        xx = torch.zeros(x.shape).to(device)
        yy = torch.zeros(y.shape).to(device)
        #print('x',x.shape)
        #print('y',y.shape)
        for i in range(x.shape[0]):
            xx[i,0] = x[i,0]-torch.mean(x[i,0])
            xx[i,1] = x[i,1]-torch.mean(x[i,1])
            xx[i,2] = x[i,2]-torch.mean(x[i,2])
            yy[i,0] = y[i,0]-torch.mean(y[i,0])
            yy[i,1] = y[i,1]-torch.mean(y[i,1])
            yy[i,2] = y[i,2]-torch.mean(y[i,2])
        for i in range(x.shape[0]):
            a=torch.zeros([1, xx.shape[1], xx.shape[2], xx.shape[3]]).to(device)
            a[0]= xx[i,:,:,:]
            #print('a', a.shape)
            b=torch.zeros([1, xx.shape[1], xx.shape[2], xx.shape[3]]).to(device)
            b[0]= yy[i,:,:,:]
            #print('b',b.shape)
            conv = F.conv2d(a, torch.transpose(b,0,1), padding = int(xx.shape[2]/2), groups=3).to(device)
            #print('conv', conv.shape)
            conv2 = conv[:,:,:xx.shape[2],:xx.shape[2]].to(device)
            #print('conv2', conv2.shape)
            #print('cati', cat[i].shape)
            #print('cat', cat.shape)
            cat[i] = conv2[0]
        
        #print('cat', cat.shape)
        #print('x', x.shape)
        x3 = self.convnetcorr(cat)
        x3 = x3.view(x.size(0), 256*6*6)
        """
        x1 = self.convnet(x)
        x1 = x1.view(x.size(0), 512*7*7)
        x2 = self.convnet(y)
        x2 = x2.view(x.size(0), 512*7*7)
        
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x
