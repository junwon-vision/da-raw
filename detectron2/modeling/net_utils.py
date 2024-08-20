import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.functional import normalize
from torchvision.transforms import GaussianBlur
import fvcore.nn.weight_init as weight_init

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)



class domain_pixel_cls(nn.Module):
    def __init__(self, dim=1024, k_size = 3):
        super(domain_pixel_cls, self).__init__()
        self.conv1= nn.Conv2d(in_channels=dim, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        # x = x.view(-1)
        return x



class domain_img_cls(nn.Module):
    def __init__(self, dim=1024, k_size = 5):
        super(domain_img_cls, self).__init__()
        self.conv1 = conv3x3(dim, 256, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        # self.avg_pool = nn.AvgPool2d(kernel_size=pooler_resolution, stride=pooler_resolution)
        self.fc = nn.Linear(128,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, x):
        # 
        # x = self.gaussianblur(x)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        
        x = x.view(-1,128)
        
        x = self.sigmoid(self.fc(x))
        return x
    

def create_gram_feat(input):
    b, c, h, w = input.size()
    gram_list = [0]*b
    for i in range(b):
        feat = input[i].view(c, h*w)
        gram = torch.mm(feat, feat.t())
        gram_feat = gram[torch.triu(torch.ones(gram.size()[0], gram.size()[1])) == 1]
        gram_feat.requires_grad_(True)
        gram_list[i] = gram_feat
    # 
    gram_feat = torch.stack(gram_list)
    # gram_feat_dict[layer] = gram_list
    
    return gram_feat

 

class domain_img_cls_GRAM(nn.Module):
    def __init__(self, dim=1024):
        super(domain_img_cls_GRAM, self).__init__()
        
        gram_dim = int(dim*(dim+1)/2)
        # self.bn0 = nn.BatchNorm1d(gram_dim)
        self.fc1 = nn.Linear(gram_dim, gram_dim//2) # number of hidden layer
        # self.bn1 = nn.BatchNorm1d(gram_dim//2)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(gram_dim//2, gram_dim//4)
        # self.bn2 = nn.BatchNorm1d(gram_dim//4)
        self.fc3 = nn.Linear(gram_dim//4,64)
        # self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.normalize = normalize
        
    def forward(self, x):
        # gram matrix
        # 
        x = create_gram_feat(x)
        # x = self.bn0(x)
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = normalize(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    

class domain_img_cls_GRAM_thin(nn.Module):
    def __init__(self, dim=1024):
        super(domain_img_cls_GRAM_thin, self).__init__()
        
        gram_dim = int(dim*(dim+1)/2)
        self.fc1 = nn.Linear(gram_dim, gram_dim//8) # number of hidden layer
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(gram_dim//8, 64)
        self.fc4 = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # gram matrix
        
        x = create_gram_feat(x)
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = normalize(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    
    

''' 
InstanceLayers:
    Discriminator for instance feature.
    InstanceLayer: input dim -> 2 fc layer output
'''    

class domain_inst_cls(nn.Module):
    def __init__(self, dim=2048):
        super(domain_inst_cls, self).__init__()
        self.fc1 = nn.Linear(dim, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout(p=0.5)
        
        # self.classifier = nn.Linear(1024,2)
        self.classifier = nn.Linear(1024,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        
        x = self.sigmoid(self.classifier(x))
        x = x.view(-1)
        return x
    
    

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    

class AdaptiveReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
    

class ScaleLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha

        return output, None


class ContrastiveHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
        
    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized