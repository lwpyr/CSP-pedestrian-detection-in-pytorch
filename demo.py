from __future__ import division
import torch
import cv2
from torchvision.transforms import ToTensor,Normalize,Compose

from config import Config
from net.network import CSPNet
from util.functions import *




config = Config()
config.offset = True
config.size_test = (640, 1280)

net = CSPNet().cuda()
net.load_state_dict(torch.load('./ckpt/CSPNet-1.pth'))

tran = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

filepath = './000007.jpg'
img = cv2.imread(filepath)
rimg, scale = resize(img, *config.size_test)
img = rimg.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inputs = tran(img).cuda()

# define the network prediction
with torch.no_grad():
    pos, height, offset = net(inputs.unsqueeze(dim=0))

boxes = parse_det_offset(pos.cpu(), height.cpu(), offset.cpu(), config.size_test, score=0.1, down=4, nms_thresh=0.3)
vis_detections(rimg, boxes, './test.jpg')
