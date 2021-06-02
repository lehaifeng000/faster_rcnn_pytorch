import os

import torch
# os.chdir(os.path.split(os.path.realpath(__file__))[0])

# from torchvision.models.detection import fasterrcnn_resnet50_fpn
from net.faster_rcnn import  fasterrcnn_resnet50
import cv2
from torchvision import transforms
from torch import optim
from tqdm import tqdm
import numpy as np
from collections import deque

from util.pkl_util import save_obj
from util.cal_map import calculate_VOC_mAP


def fit_one_step(model,optimizer,train_loader, val_loader,epoch=0):
    tqdm_bar=tqdm(train_loader)
    max_len=50
    all_loss,loss_classifier,loss_box_reg,loss_objectness,loss_rpn_box_reg=deque(maxlen=max_len),deque(maxlen=max_len),deque(maxlen=max_len),deque(maxlen=max_len),deque(maxlen=max_len),
    targets = []
    images = []
    model.train()
    # for step_id, (imgs,bboxs,labels) in enumerate(tqdm_bar):
    for step_id, (images, targets, _) in enumerate(tqdm_bar):
        # if step_id>20:
        #     break
        optimizer.zero_grad()
        ret = model(images,targets)
        loss = sum(ret.values())
        loss.backward()
        optimizer.step()
        # all_loss.append(sum(ret.values()).cpu().detach().numpy())
        # loss_classifier.append(ret['loss_classifier'].cpu().detach().numpy())
        # loss_box_reg.append(ret['loss_box_reg'].cpu().detach().numpy())
        # loss_objectness.append(ret['loss_objectness'].cpu().detach().numpy())
        # loss_rpn_box_reg.append(ret['loss_rpn_box_reg'].cpu().detach().numpy())

        all_loss.append(sum(ret.values()).item())
        loss_classifier.append(ret['loss_classifier'].item())
        loss_box_reg.append(ret['loss_box_reg'].cpu().item())
        loss_objectness.append(ret['loss_objectness'].item())
        loss_rpn_box_reg.append(ret['loss_rpn_box_reg'].item())
        ml=min(step_id+1,max_len)
        tqdm_bar.set_description("epoch:{: ^4} loss_classifier:{:.4f}  loss_box_reg:{:.4f}  loss_objectness:{:.4f}  loss_rpn_box_reg:{:.4f}  total_loss:{:.4f}".format(
                epoch+1,
                sum(loss_classifier)/ml,
                sum(loss_box_reg)/ml,
                sum(loss_objectness)/ml,
                sum(loss_rpn_box_reg)/ml,
                sum(all_loss)/ml
            ))
        tqdm_bar.update(1)
        pass


def unfreeze(model):
    # t=model.backbone.parameters()
    for param in model.backbone.parameters():
            param.requires_grad = True

def validation(model, val_loader):
    tqdm_bar=tqdm(val_loader)
    model.eval()
    results = []
    # with torch.no_grad():
    with torch.no_grad():
        for step_id, (images, targets, img_ids) in enumerate(tqdm_bar):
            # if step_id>10:
            #     break
            ret = model(images)
            for idx in range(len(img_ids)):
                pre_boxes=ret[idx]['boxes'].cpu().numpy()
                pre_labels=ret[idx]['labels'].cpu().numpy()
                pre_scores=ret[idx]['scores'].cpu().numpy()
                gt_boxes=targets[idx]['boxes'].cpu().numpy()
                gt_labels=targets[idx]['labels'].cpu().numpy()
                results.append((img_ids[idx],pre_boxes,pre_labels,pre_scores,gt_boxes,gt_labels))
            pass
        # save_obj(results,'results.pkl')
        print(calculate_VOC_mAP(results, iou_thresh=0.5)[0])
        pass


model = fasterrcnn_resnet50(pretrained_backbone=True,num_classes=21, trainable_backbone_layers=1,min_size=600, max_size=800,)#  fasterrcnn_resnet50_fpn, fasterrcnn_resnet50 pretrained=True

model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)   # , weight_decay=1e-5
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
from util.voc_util import get_loader

BATCH_SIZE = 1

train_loader = get_loader(batch_size=BATCH_SIZE)
val_loader = get_loader(is_train=False, batch_size=BATCH_SIZE)
# model.train()
# torch.save(model.state_dict(), 'parameter.pkl')
model.load_state_dict(torch.load('parameter.pkl'))
EPOCHS = 30
unfreeze(model)
for epoch in range(EPOCHS):
    if epoch==EPOCHS//2:
        unfreeze(model)
    fit_one_step(model,optimizer, train_loader,val_loader,epoch)
    torch.save(model.state_dict(), 'parameter.pkl')
    lr_scheduler.step()
    validation(model, val_loader)
# validation(model, val_loader)





# img = cv2.imread('../imgs/002484.jpg')
# img = transforms.ToTensor()(img)
# img=img.unsqueeze(0)

# model.eval()
# ret = model(img)



pass