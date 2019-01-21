import os

import torch
import torch.optim as optim
from torch import nn

import refs
import adda_utils

def print_step(epoch, step, max_step, disc_loss, tgt_loss, disc_acc):
      msg = "Epoch (%d / %d), Step: (%d / %d): disc_loss: %.5f" \
      "| tgt_loss: %.5f | disc_acc: %.5f" % (epoch, refs.adda_epochs, 
                                             step, max_step, disc_loss, 
                                             tgt_loss, disc_acc)
      print(msg)

def save_state(tgt_model, disc_model, sufix, root=refs.adda_models_root_path):
      discriminator_filename = 'discriminator_model_%s.pt' % (sufix)
      discriminator_filename = os.path.join(root, discriminator_filename)
      torch.save(disc_model.state_dict(), discriminator_filename)
      
      target_filename = 'target_model_%s.pt' % (sufix)
      target_filename = os.path.join(root, target_filename)
      torch.save(tgt_model.state_dict(), target_filename)

      print('Target and Discriminator models saved at: %s | %s' % (
                  target_filename, discriminator_filename))

def train(src_model, tgt_model, discriminator, 
          src_data_loader, tgt_data_loader):
      # Define optimizers, loss and setup models for training:
      tgt_optimizer = optim.Adam(tgt_model.parameters(), 
                                 lr=refs.tgt_lr,
                                 betas=(refs.beta1, refs.beta2))
      discriminator_optimizer = optim.Adam(discriminator.parameters(), 
                                           lr=refs.adda_discriminator_lr,
                                           betas=(refs.beta1, refs.beta2))
      criterion = nn.CrossEntropyLoss()
      tgt_model.train()
      discriminator.train()
      # Loop over epochs
      for epoch in range(refs.adda_epochs):
            data_loader = zip(src_data_loader, tgt_data_loader)
            n_steps = min(len(src_data_loader), len(tgt_data_loader))
            # Loop over dataset
            for step, data in enumerate(data_loader):
                  (src_images, _), (tgt_images, _) = data
                  
                  # Handle device & converto to variables imgs_src and imgs_tgt
                  src_images = adda_utils.make_variable(src_images)
                  tgt_images = adda_utils.make_variable(tgt_images)
                  
                  # Update discriminator
                  src_features = src_model.Encode(src_images)
                  tgt_features = tgt_model.Encode(tgt_images)
                  concat_features = torch.cat((src_features, tgt_features), 0)
                  
                  # No need for gradient computation, since we are going to
                  # to update the discriminator (thus no gradients from feature
                  # extraction are necessary)
                  domain_predictions = discriminator(concat_features.detach()) 
                  src_labels = adda_utils.make_variable(torch.ones(
                              src_features.size(0)).long())
                  tgt_labels = adda_utils.make_variable(torch.zeros(
                              tgt_features.size(0)).long())
                  discriminator_optimizer.zero_grad()
                  labels = torch.cat((src_labels, tgt_labels), 0)
                  disc_loss = criterion(domain_predictions, labels)
                  disc_loss.backward()
                  discriminator_optimizer.step()
                  pred_cls = torch.squeeze(domain_predictions.max(1)[1])
                  disc_acc = (pred_cls == labels).float().mean()
                  #disc_acc = float(pred_cls.eq(labels.data).cpu().sum()) / float(refs.batch_size)
                  
                  """ Question: Could we somehow avoid to forwad pass once more 
                  the tgt_images to the tgt_model?"""
                  
                  #-- update tgt_model with flipped loss
                  discriminator_optimizer.zero_grad() # I don't think we need this
                  tgt_optimizer.zero_grad()
                  
                  # Do we really need to pass forward once more here?
                  tgt_features = tgt_model.Encode(tgt_images)
                  domain_predictions = discriminator(tgt_features)
                  labels = adda_utils.make_variable(torch.ones(
                              tgt_features.size(0)).long())
                  tgt_loss = criterion(domain_predictions, labels)
                  tgt_loss.backward()
                  tgt_optimizer.step()
                  #Print step info
                  if step % refs.adda_step_log == 0:
                        print_step(epoch, step, n_steps, disc_loss.data[0], tgt_loss.data[0], disc_acc.data[0])
            #Print stuff
            if epoch % refs.adda_save_each_epoch == 0 and epoch > 0:
                  #Save model
                  save_state(tgt_model, discriminator, '%d'%(epoch))
      save_state(tgt_model, discriminator, 'final')