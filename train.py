from __future__ import print_function

import os

import torch
import torch.optim as optim
from torch import nn

import adda_utils
import refs



def print_step(epoch, step, max_step, loss, acc):
      msg = 'Epoch (%d / %d); Step (%d / %d); loss: %.5f | on_batch_acc: %.5f' % (epoch,
                   refs.src_epochs, step, max_step, loss, acc)
      print(msg)



def save_model(model, root, sufix):
      filename = 'source_model_%s.pt' % (sufix)
      filename = os.path.join(root, filename)
      torch.save(model.state_dict(), filename)
      print('Saved model at: %s' % (filename))



def train(model, data_loader, root=refs.src_exp_root):
      # Set optimizer and criterion
      optmizer = optim.Adam(model.parameters(), lr=refs.src_lr, 
                            betas=(refs.beta1, refs.beta2))
      criterion = nn.CrossEntropyLoss()
      model.train()
      # Loop over epochs
      for epoch in range(refs.src_epochs):
            # Loop over steps
            max_step = len(data_loader)
            for step, (src_images, labels) in enumerate(data_loader):
                  # Handle device type:
                  src_images = adda_utils.make_variable(src_images)
                  labels = adda_utils.make_variable(labels.squeeze_())
                  # Optmize model
                  optmizer.zero_grad()
                  predicted_labels = model(src_images)
                  loss = 0.
                  acc = 0.
                  loss = criterion(predicted_labels, labels)
                  loss.backward()
                  optmizer.step()
                  if step % refs.src_step_log == 0 and step > 0:
                        pred_cls = predicted_labels.data.max(1)[1]
                        acc = float(pred_cls.eq(labels.data).cpu().sum()) / float(refs.batch_size)
                        print_step(epoch, step, max_step, loss.data[0], acc)
            if epoch % refs.src_epoch_log == 0 and epoch > 0:
                  save_model(model, root, '%d' % (epoch))
      save_model(model, root, 'final')
      return model



def evaluate(model, data_loader):
      model.eval()
      criterion = nn.CrossEntropyLoss()
      loss = 0.
      acc = 0.
      for (src_images, labels) in data_loader:
            src_images = adda_utils.make_variable(src_images, volatile=True)
            labels = adda_utils.make_variable(labels).squeeze_()
            predictions = model(src_images)
            loss += criterion(predictions, labels).data[0]
            pred_class = predictions.data.max(1)[1]
            acc += pred_class.eq(labels.data).cpu().sum()
      acc = float(acc)/float(len(data_loader.dataset))
      loss /= len(data_loader)
      print('Evaluation done: Loss: %.5f | Accuracy: %f' % (loss, acc))
      return loss, acc
