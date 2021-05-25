import os
# os.chdir(os.path.split(os.path.realpath(__file__))[0])

# from torchvision.models.detection import fasterrcnn_resnet50_fpn
from net.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50
import cv2
from torchvision import transforms
from torch import optim
from tqdm import tqdm




def fit_one_step(model,optimizer,train_loader, val_loader,epoch=0):
    tqdm_bar=tqdm(train_loader)
    all_loss=[]
    targets = []
    images = []
    # for step_id, (imgs,bboxs,labels) in enumerate(tqdm_bar):
    for step_id, (images, targets) in enumerate(tqdm_bar):
        # for bindx in range(len(labels)):
        #     d={}
        #     d['boxes'] = bboxs[bindx].cuda()
        #     d['labels'] = labels[bindx].cuda()
        #     targets.append(d)
        #     images.append(imgs[bindx].cuda())
        # if step_id==0:
        #     continue
        # if step_id==5:
        #     a=1

        ret = model(images,targets)
        loss = sum(ret.values())
        loss.backward()
        optimizer.step()
        all_loss.append(sum(ret.values()).cpu().detach().numpy())
        tqdm_bar.set_description("epoch:{: ^4} loss_classifier:{:.4f}  loss_box_reg:{:.4f}  loss_objectness:{:.4f}  loss_rpn_box_reg:{:.4f}  total_loss:{:.4f}".format(
                epoch+1,
                ret['loss_classifier'].cpu().detach().numpy(),
                ret['loss_box_reg'].cpu().detach().numpy(),
                ret['loss_objectness'].cpu().detach().numpy(),
                ret['loss_rpn_box_reg'].cpu().detach().numpy(),
                sum(all_loss)/len(all_loss)
            ))
        tqdm_bar.update(1)
        pass




model = fasterrcnn_resnet50(pretrained_backbone=True)# fasterrcnn_resnet50_fpn, fasterrcnn_resnet50

model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)

from util.voc_util import get_loader

train_loader = get_loader(batch_size=4)
val_loader = get_loader(is_train=False)

EPOCHS = 5
for epoch in range(EPOCHS):
    fit_one_step(model,optimizer, train_loader,val_loader,epoch)





# img = cv2.imread('../imgs/002484.jpg')
# img = transforms.ToTensor()(img)
# img=img.unsqueeze(0)

# model.eval()
# ret = model(img)



pass