import os
# os.chdir(os.path.split(os.path.realpath(__file__))[0])

# from torchvision.models.detection import fasterrcnn_resnet50_fpn
from net.faster_rcnn import fasterrcnn_resnet50_fpn
import cv2
from torchvision import transforms
from torch import optim
from tqdm import tqdm




def fit_one_step(model,optimizer,train_loader, val_loader,epoch=0):
    # tqdm_bar=tqdm(train_loader)
    all_loss=[]
    # targets = []
    # images = []
    # for step_id, (imgs,bboxs,labels) in enumerate(tqdm_bar):
    # for step_id, (imgs,bboxs,labels) in enumerate(train_loader):
    inputs = train_loader.get_batch()
    while inputs:
        images, targets=inputs

        ret = model(images,targets)
        loss = sum(ret.values())
        loss.backward()
        optimizer.step()
        all_loss.append(sum(ret.values()).cpu().detach().numpy())
        # tqdm_bar.set_description("epoch:{: ^4} loss_classifier:{:.4f}  loss_box_reg:{:.4f}  loss_objectness:{:.4f}  loss_rpn_box_reg:{:.4f}  total_loss:{:.4f}".format(
        #         epoch+1,
        #         ret['loss_classifier'].cpu().detach().numpy(),
        #         ret['loss_box_reg'].cpu().detach().numpy(),
        #         ret['loss_objectness'].cpu().detach().numpy(),
        #         ret['loss_rpn_box_reg'].cpu().detach().numpy(),
        #         sum(all_loss)/len(all_loss)
        #     ))
        # del ret
        inputs = train_loader.get_batch()
        pass




model = fasterrcnn_resnet50_fpn(pretrained=True)

model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)

from util.voc_data import FRCNNDataset

train_loader = FRCNNDataset()
# val_loader = get_loader(is_train=False)

EPOCHS = 5
for epoch in range(EPOCHS):
    fit_one_step(model,optimizer, train_loader,train_loader,epoch)





# img = cv2.imread('../imgs/002484.jpg')
# img = transforms.ToTensor()(img)
# img=img.unsqueeze(0)

# model.eval()
# ret = model(img)



pass