import os
import time
import torch
import json
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network import CSPNet, CSPNet_mod
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate


config = Config()
config.train_path = './data/citypersons'
config.test_path = './data/citypersons'
config.gpu_ids = [0, 1]
config.onegpu = 4
config.size_train = (640, 1280)
config.size_test = (1024, 2048)
config.init_lr = 2e-4
config.num_epochs = 150
config.offset = True
config.val = True
config.val_frequency = 1

# dataset
print('Dataset...')
traintransform = Compose(
    [ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
traindataset = CityPersons(path=config.train_path, type='train', config=config,
                           transform=traintransform)
trainloader = DataLoader(traindataset, batch_size=config.onegpu*len(config.gpu_ids))

if config.val:
    testtransform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testdataset = CityPersons(path=config.train_path, type='val', config=config,
                              transform=testtransform, preloaded=True)
    testloader = DataLoader(testdataset, batch_size=1)

# net
print('Net...')
net = CSPNet().cuda()
# To continue training
#net.load_state_dict(torch.load('./ckpt/DataParallel-9.pth'))

# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()

# optimizer
params = []
for n, p in net.named_parameters():
    if p.requires_grad:
        params.append({'params': p})
    else:
        print(n)

if config.teacher:
    print('I found this teacher model is useless, I disable this training option')
    exit(1)
    teacher_dict = net.state_dict()

#if len(config.gpu_ids) > 1:
net = nn.DataParallel(net, device_ids=config.gpu_ids)

optimizer = optim.Adam(params, lr=config.init_lr)


batchsize = config.onegpu * len(config.gpu_ids)
train_batches = len(trainloader)

config.print_conf()


def criterion(output, label):
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    return cls_loss, reg_loss, off_loss


def train():

    print('Training start')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # open log file
    log_file = './log/' + time.strftime('%Y%m%d', time.localtime(time.time()))+'.log'
    log = open(log_file, 'w')
    if config.val:
        vallog_file = log_file + '.val'
        vallog = open(vallog_file, 'w')

    best_loss = np.Inf
    best_loss_epoch = 0

    best_mr = 100
    best_mr_epoch = 0

    for epoch in range(150):
        print('----------')
        print('Epoch %d begin' % (epoch + 1))
        t1 = time.time()

        epoch_loss = 0.0
        net.train()

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
            if config.teacher:
                for k, v in net.module.state_dict().items():
                    if k.find('num_batches_tracked') == -1:
                        teacher_dict[k] = config.alpha * teacher_dict[k] + (1 - config.alpha) * v
                    else:
                        teacher_dict[k] = 1 * v

            # print statistics
            batch_loss = loss.item()
            batch_cls_loss = cls_loss.item()
            batch_reg_loss = reg_loss.item()
            batch_off_loss = off_loss.item()

            t4 = time.time()
            print('\r[Epoch %d/150, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, Time: %.3f sec        ' %
                  (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss, t4-t3)),
            epoch_loss += batch_loss
        print('')

        t2 = time.time()
        epoch_loss /= len(trainloader)
        print('Epoch %d end, AvgLoss is %.6f, Time used %.1f sec.' % (epoch+1, epoch_loss, int(t2-t1)))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_epoch = epoch + 1
        print('Epoch %d has lowest loss: %.7f' % (best_loss_epoch, best_loss))

        if config.val and epoch + 1 > 10 and (epoch + 1) % config.val_frequency == 0:
            cur_mr = val(vallog)
            if cur_mr < best_mr:
                best_mr = cur_mr
                best_mr_epoch = epoch + 1
            print('Epoch %d has lowest MR: %.7f' % (best_mr_epoch, best_mr))

        log.write('%d %.7f\n' % (epoch+1, epoch_loss))
            
        print('Save checkpoint...')
        filename = './ckpt/%s-%d.pth' % (net.module.__class__.__name__, epoch+1)

        torch.save(net.module.state_dict(), filename)
        if config.teacher:
            torch.save(teacher_dict, filename+'.tea')

        print('%s saved.' % filename)

    log.close()
    if config.val:
        vallog.close()


def val(log=None):
    net.eval()

    if config.teacher:
        print('Load teacher params')
        student_dict = net.module.state_dict()
        net.module.load_state_dict(teacher_dict)

    print('Perform validation...')
    res = []
    t3 = time.time()
    for i, data in enumerate(testloader, 0):
        inputs = data.cuda()
        with torch.no_grad():
            pos, height, offset = net(inputs)

        boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

            for box in boxes:
                temp = dict()
                temp['image_id'] = i+1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)

        print('\r%d/%d' % (i + 1, len(testloader))),
        sys.stdout.flush()
    print('')

    if config.teacher:
        print('Load back student params')
        net.module.load_state_dict(student_dict)

    with open('./_temp_val.json', 'w') as f:
        json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', './_temp_val.json')
    t4 = time.time()
    print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    if log is not None:
        log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs[0]


if __name__ == '__main__':
    train()
    #val()
