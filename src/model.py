import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import math 

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

        #CHOSE THE PRETRAINED MODEL YOU WANT TO USE (UNCOMMENT IT)
        #caffenet = models.alexnet(pretrained=True) #(256*6*6)
        #caffenet = models.vgg16(pretrained=True) #(7*7*512)
        #caffenet = models.shufflenet_v2_x0_5(pretrained=True) #(1024*7*7)
        #caffenet = models.mobilenet_v2(pretrained=True) # (7*7*1280)
        caffenet = models.mnasnet0_5(pretrained = True) #( 7*7*1280)
        #caffenet = models.resnet18(pretrained = True) #(512*2)
        #caffenet = models.resnet50(pretrained = True) #(1024)
        self.convnet = nn.Sequential(*list(caffenet.children())[:-1])
        for param in self.convnet.parameters():
            param.requires_grad = False
        
        """ #USED WITH CORRELATION OF TWO IMAGES
        caffenetcorr = models.alexnet(pretrained=False)
        self.convnetcorr = nn.Sequential(*list(caffenetcorr.children())[:-1])
        for param in self.convnetcorr.parameters():
            param.requires_grad = True 
        """
        """
        #### GROUP CONNECTED LAYER
        size_input = 7*7*1280*2

        self.layer10 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer11 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer12 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer13 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer14 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer15 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer16 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer17 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer18 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer19 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer110 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer111 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer112 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer113 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer114 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer115 = nn.Linear(math.ceil(size_input/20), 205, bias=True)
        self.layer116 = nn.Linear(math.ceil(size_input/20), 204, bias=True)
        self.layer117 = nn.Linear(math.ceil(size_input/20), 204, bias=True)
        self.layer118 = nn.Linear(math.ceil(size_input/20), 204, bias=True)
        self.layer119 = nn.Linear(math.ceil(size_input/20), 204, bias=True)

        self.classifier = nn.Sequential(
                #nn.Linear(256*6*6*2, 4096),
                #nn.Linear(256*6*6*2, 4096), #
                #nn.Linear(512*7*7*2, 4096), #
                #nn.Linear(1280*7*7*2, 4096), #mobilenet
                #nn.Linear(7*7*1280*2,4096), #mnasnet
                nn.Linear(4096, 4096), 
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

        """

        #UNCOMMENT DEPENDING ON THE PRETRAINED NETWORK YOU CHOSE
        self.classifier = nn.Sequential(
                #nn.Linear(256*6*6*2, 4096), #ALEXNET
                #nn.Linear(512*7*7*2, 4096), #VGG16
                #nn.Linear(1280*7*7*2, 4096), #MOBILENET
                nn.Linear(7*7*1280*2,4096), #MNASNET
                #nn.Linear(1024, 4096), #RESNET50 AND RESNET18  
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
        
        """ USED TO DO IMAGE CORRELATION
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
        #CHANGE DEPENDING ON THE OUTPUT SIZE OF THE PRETRAINED MODEL YOU USED (SEE COMMENTS ABOVE)
        x1 = x1.view(x.size(0),7*7*1280) #6*6*256) #512*7*7)  #7*7*1280)#1280*7*7) #512) #256*6*6)
        x2 = self.convnet(y)
        x2 = x2.view(x.size(0), 7*7*1280) #6*6*256) #512*7*7)  #1280*7*7) #256*6*6)

        x= torch.cat((x1,x2),1)
        
        
        """used to experiment if "fully connected layer grouping" would work (sadly not)
        #My creation
        #features = torch.cat((x1, x2), 1)

        sz = 7*7*1280*2
        szz = 6272 
           
        x10 = self.layer10(features[: , 0 : szz])          
        x11 = self.layer11(features[: , szz : 2*szz])          
        x12 = self.layer12(features[: , 2*szz : 3*szz])          
        x13 = self.layer13(features[: , 3*szz : 4*szz])          
        x14 = self.layer14(features[: , 4*szz : 5*szz])          
        x15 = self.layer15(features[: , 5*szz : 6*szz])          
        x16 = self.layer16(features[: , 6*szz : 7*szz])          
        x17 = self.layer17(features[: , 7*szz : 8*szz])          
        x18 = self.layer18(features[: , 8*szz : 9*szz ])          
        x19 = self.layer19(features[: , 9*szz : 10*szz ])          
        x110 = self.layer110(features[: , 10*szz : 11*szz])          
        x111 = self.layer111(features[: , 11*szz : 12*szz])          
        x112 = self.layer112(features[: , 12*szz : 13*szz])          
        x113 = self.layer113(features[: , 13*szz : 14*szz])          
        x114 = self.layer114(features[: , 14*szz : 15*szz])          
        x115 = self.layer115(features[: , 15*szz : 16*szz])          
        x116 = self.layer116(features[: , 16*szz : 17*szz]) 
        x117 = self.layer117(features[: , 17*szz : 18*szz]) 
        x118 = self.layer118(features[: , 18*szz : 19*szz])  
        x119 = self.layer119(features[: , 19*szz : 20*szz])   
        x = torch.cat((x10,x11,x12,x13,x14,x15,x16,x17,x18,x19, x110,x111,x112,x113,x114,x115,x116,x117,x118,x119), 1)
        """


        x = self.classifier(x)
        return x
