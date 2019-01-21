import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import lenet_model

"""
Separetes Feature extraction from Prediction:
      ADDA_models implements the following api:
            Encode(self, x): outputs encoding of x according to the model
            Predict(self, x): predict the class of x.
"""

class ADDAResNet(models.resnet.ResNet):

      def __init__(self, block, layers, num_classes=1000):
            super(ADDAResNet, self).__init__(block, layers, num_classes)
            
            # Freezes classifier
            for name, param in self.named_parameters():
                  if name in ['fc']:
                        param.requires_grad = False

      def Encode(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            return x

      def Predict(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)

################### ADDAResNet helper constructors ############################
def resnet18(num_classes=1000):
    model = ADDAResNet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes)
    return model

def resnet34(num_classes=1000):
    model = ADDAResNet(models.resnet.BasicBlock, [3, 4, 6, 3], num_classes)
    return model

def resnet50(num_classes=1000):
    model = ADDAResNet(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes)
    return model

def resnet101(num_classes=1000):
    model = ADDAResNet(models.resnet.Bottleneck, [3, 4, 23, 3], num_classes)
    return model

def resnet152(num_classes=1000):
    model = ADDAResNet(models.resnet.Bottleneck, [3, 8, 36, 3], num_classes)
    return model
###############################################################################
    

class ADDALeNet(lenet_model.LeNet):
      def __init__(self):
            super(ADDALeNet, self).__init__()
                        # Freezes classifier
            #for name, param in self.named_parameters():
                  #if name in ['fc2']:
                        #param.requires_grad = False
                  
            
      def Encode(self, x):
            conv_out = self.encoder(x)
            feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
            return feat
      
      def Predict(self, x):
            feat = F.dropout(F.relu(x), training=self.training)
            feat = self.fc2(feat)
            return feat
      
      def copyFrom(self, model):
            self.load_state_dict(model.state_dict())


class ADDADiscriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(ADDADiscriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out