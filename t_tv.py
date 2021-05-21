import os
# os.chdir(os.path.split(os.path.realpath(__file__))[0])

# from torchvision.models.detection import fasterrcnn_resnet50_fpn
from net.faster_rcnn import fasterrcnn_resnet50_fpn
import cv2
from torchvision import transforms
from tqdm import tqdm




def fit_one_step(model,train_loader, val_loader,epoch=0):
    tqdm_bar=tqdm(train_loader)
    all_loss=[]
    for i, (imgs,bboxs,labels) in enumerate(tqdm_bar):
        targets = []
        d={}
        d['boxes'] = bboxs[0]
        d['labels'] = labels[0]
        targets.append(d)
        images = list(image for image in imgs)
        ret = model(images,targets)
        all_loss.append(sum(ret.values()))
        tqdm_bar.set_description("epoch:{: ^4} loss_classifier:{:.4f}  loss_box_reg:{:.4f}  loss_objectness:{:.4f}  loss_rpn_box_reg:{:.4f}  total_loss:{:.4f}".format(
                epoch+1,
                ret['loss_classifier'],
                ret['loss_box_reg'],
                ret['loss_objectness'],
                ret['loss_rpn_box_reg'],
                sum(all_loss)/len(all_loss)
            ))
        pass




model = fasterrcnn_resnet50_fpn(pretrained=False)

from util.voc_util import get_loader

train_loader = get_loader()
val_loader = get_loader(is_train=False)

EPOCHS = 5
for epoch in range(EPOCHS):
    fit_one_step(model,train_loader,val_loader,epoch)





# img = cv2.imread('../imgs/002484.jpg')
# img = transforms.ToTensor()(img)
# img=img.unsqueeze(0)

# model.eval()
# ret = model(img)



pass