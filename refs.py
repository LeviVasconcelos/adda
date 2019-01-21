tgt_lr = 1e-4

adda_discriminator_lr=1e-4
adda_discriminator_betas=3

adda_epochs=2000
adda_save_each_epoch=100
adda_step_log=100
adda_models_root_path='exp/adda_torch_020_from_lenet'

#Params for source training
src_epochs = 100
src_exp_root = 'exp/src_train100/'
src_lr = 1e-4
src_epoch_log = 50
src_step_log = 13
src_betas = 0.5

#Params for mnist dataset
mnist_root = 'data/mnist'
usps_root = 'data/usps'

############################
"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64

# params for source dataset
src_dataset = "MNIST"
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = False

# params for target dataset
tgt_dataset = "USPS"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = False

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 100
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 2000
log_step = 100
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9