from models import lenet_model, adda_models
from datasets import mnist, usps
from train import train, evaluate
import adda_utils
import adda_train

src_model = adda_utils.init_model(net=adda_models.ADDALeNet(), 
                                       restore='exp/from_lenet_model/source_model_50.pt')
tgt_model = adda_utils.init_model(net=adda_models.ADDALeNet(), 
                                       restore='exp/from_lenet_model/source_model_50.pt')

#src_model = adda_models.ADDALeNet()
#src_model.copyFrom(src_lenetmodel)
#tgt_model = adda_models.ADDALeNet()
#tgt_model.copyFrom(src_lenetmodel)

src_model = adda_utils.make_cuda(src_model)
tgt_model = adda_utils.make_cuda(tgt_model)

src_data_loader = mnist.get_mnist(train=True)
src_data_loader_eval = mnist.get_mnist(train=False)
tgt_data_loader = usps.get_usps(train=True)
tgt_data_loader_eval = usps.get_usps(train=False)

discriminator =  adda_utils.init_model(adda_models.ADDADiscriminator(500, 500, 2), None)

adda_train.train(src_model, tgt_model, discriminator, 
                 src_data_loader, tgt_data_loader)