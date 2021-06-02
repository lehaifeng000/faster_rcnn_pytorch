import torch
import cv2
from torchvision import transforms
from PIL import Image

# img = cv2.imread('./imgs/000000133933.jpg')
img = Image.open('./imgs/000000133933.jpg')
img = transforms.ToTensor()(img).cuda()


toPIL = transforms.ToPILImage()
pic = toPIL(img)
pic.save('t1.jpg')

pass