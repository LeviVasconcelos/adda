from models import lenet_model, adda_models
from datasets import mnist, usps
from train import train, evaluate
import adda_utils


#model = lenet_model.LeNet()
#model = 
model = adda_utils.init_model(net=lenet_model.LeNet(), restore='exp/adda_torch_020/target_model_100.pt')
#data_loader = mnist.get_mnist(train=False)
data_loader = usps.get_usps(train=False)
evaluate(model, data_loader)
#trained_model = train(model, data_loader, 'exp/from_lenet_model')