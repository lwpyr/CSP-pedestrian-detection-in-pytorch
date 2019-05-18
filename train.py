from net.loss import *
from net.network import CSPNet
from config import Config
from dataloader.loader import *
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

import torch
import os
import time
import torch.optim as optim
from matplotlib import pyplot as plt


config = Config()
config.train_path = '/data/liwen/dataset'
config.gpu_ids = '0'
config.onegpu = 4
config.size_train = (640, 1280)
config.init_lr = 2e-4
config.num_epochs = 150
config.offset = True

# dataset
print('Dataset...')
transform = Compose([ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = CityPersons(path=config.train_path, train='train', config=config,
                      transform=transform)
trainloader = DataLoader(dataset, batch_size=config.onegpu)
print(len(dataset))

# net
print('Net...')
net = CSPNet().cuda()
# To continue training
# net.load_state_dict(torch.load('./ckpt/CSPNet-1.pth'))

# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()

# optimizer
params = []
for p in net.parameters():
    if p.requires_grad:
        params.append({'params': p})
optimizer = optim.Adam(params, lr=config.init_lr)


def criterion(output, label):
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    return cls_loss, reg_loss, off_loss


def train():
    batchsize = config.onegpu * len(config.gpu_ids.split(','))
    print('Training start')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    for epoch in range(150):
        t1 = time.time()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            t3 = time.time()
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = [l.cuda().float() for l in labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # heat map
            outputs = net(inputs)

            # loss
            cls_loss, reg_loss, off_loss = criterion(outputs, labels)
            loss = cls_loss + reg_loss + off_loss

            # back-prop
            loss.backward()

            # update param
            optimizer.step()

            # print statistics
            batch_loss = loss.item() / batchsize
            batch_cls_loss = cls_loss.item() / batchsize
            batch_reg_loss = reg_loss.item() / batchsize
            batch_off_loss = off_loss.item() / batchsize
            t4 = time.time()
            print('[Epoch %d, Batch %d]$ {1: {Total loss: %.3f}, 2: {cls: %.3f, reg: %.3f, off: %.3f}, 3: {Time: %.1f sec}}' %
                  (epoch + 1, i + 1, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss, t4-t3))
            epoch_loss += batch_loss * batchsize

        t2 = time.time()
        print('Epoch %d end, AvgLoss is %.3f, Time used %d sec.' % (epoch+1, epoch_loss/len(dataset), int(t2-t1)))
        print('Save checkpoint...')
        filename = './ckpt/%s-%d.pth' % (net.__class__.__name__, epoch+1)
        torch.save(net.state_dict(), filename)
        print('%s saved.' % filename)


if __name__=='__main__':
    train()
