import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
from torchvision import transforms

model = fasterrcnn_resnet50_fpn(pretrained=True)
model=model.cuda()
# from util.voc_util import get_loader

# loader = get_loader()

# for i, (imgs,bboxs,labels) in enumerate(loader):
#     # targets
#     # model
#     pass


img = cv2.imread('../imgs/000000133933.jpg')
img = transforms.ToTensor()(img).cuda()
# img=img.unsqueeze(0)   
img = [img]

model.eval()
ret = model(img)



pass