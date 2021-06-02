
from xml.dom.minidom import parse
from pathlib import Path
import cv2
import torch
from PIL import Image

VOCDEV_ROOT = Path('/root/dataset/VOCdevkit/')
VOC_ROOT = VOCDEV_ROOT.joinpath('VOC2007')

TRAIN_SPLIT_PATH = VOC_ROOT.joinpath('ImageSets','Main','train_trainval.txt')
TEST_SPLIT_PATH = VOC_ROOT.joinpath('ImageSets','Main','train_test.txt')

ANNOTATION_ROOT=VOC_ROOT.joinpath('Annotations')
IMG_ROOT = VOC_ROOT.joinpath('JPEGImages')

train_lines = TRAIN_SPLIT_PATH.open().read().strip('\n').split('\n')
test_lines = TEST_SPLIT_PATH.open().read().strip('\n').split('\n')

classes={'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader

def get_loader(batch_size=1,is_train=True):
    data = FRCNNDataset(is_train)
    loader=DataLoader(dataset=data,batch_size=batch_size,shuffle=is_train, collate_fn=my_collate_fn)
    return loader


class FRCNNDataset(Dataset):
    def __init__(self,is_train=True) -> None:
        super().__init__()
        self.bboxs=[]
        self.labels=[]
        self.imgs=[]
        if is_train:
            self.lines=train_lines
        else:
            self.lines=test_lines
        
        self.transform = transforms.ToTensor()
        self.shape = (600,600)
        self.is_train = is_train

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        line = self.lines[index]
        line = line.split(' ')[0]
        xml_path = ANNOTATION_ROOT.joinpath(line+'.xml')
        collection = parse(str(xml_path)).documentElement
        img_name = collection.getElementsByTagName("filename")[0].childNodes[0].data
        objects = collection.getElementsByTagName("object")
    
        bboxs = np.zeros((len(objects),4),dtype=np.int64)
        labels = np.zeros(len(objects), dtype=np.int64)
        for i,object in enumerate(objects):
            label_name = object.getElementsByTagName("name")[0].childNodes[0].data
            box = object.getElementsByTagName("bndbox")[0]
            xmin = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
            bboxs[i] = np.array([xmin,ymin,xmax,ymax],dtype=np.int64)
            labels[i] = classes[label_name]

        img_path = str(IMG_ROOT.joinpath(img_name))
        # if self.is_train:
        #     img,bboxs=self.get_random_data(img_path,bboxs)
        # else:
        img = cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img,bboxs=resize_keep_aspectratio(img, bboxs, self.shape)
        
        img = self.transform(img)
        return img, bboxs, labels, line

    def get_random_data(self,image_path,gt_boxes, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        # line = annotation_line.split()
        image = Image.open(image_path)
        iw, ih = image.size
        h,w = self.shape
        # box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        box = np.copy(gt_boxes)
        
            
        # resize image
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),4))
        if len(box)>0:
            np.random.shuffle(box)
            # t=box[:, [0,2]]
            # t1=t*nw/iw
            # t2=t+dx
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),4))
            box_data[:len(box)] = box

        return image_data, box_data

    def __len__(self):
        return len(self.lines)

def my_collate_fn(batch):
    imgs, bboxs, labels, line = zip(*batch)  # transposed
    images=[]
    targets=[]
    for i in range(len(batch)):
        img=imgs[i]
        bbox=torch.Tensor(bboxs[i])
        label= torch.from_numpy(labels[i])
        if True:
            img=img.cuda()
            bbox=bbox.cuda()
            label=label.cuda()
        images.append(img)
        targets.append({'boxes':bbox,'labels':label})
    # for i, l in enumerate(label):
    #     l[:, 0] = i  # add target image index for build_targets()
    # import torch
    # return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    return images,targets, line


# 图片缩放
def resize_keep_aspectratio(image_src, bbox, dst_size):
    src_h, src_w = image_src.shape[:2]
    # print(src_h, src_w)
    dst_h, dst_w = dst_size

    # 判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))

    h_, w_ = image_dst.shape[:2]
    # print(h_, w_)

    top = int((dst_h - h_) / 2);
    down = int((dst_h - h_ + 1) / 2);
    left = int((dst_w - w_) / 2);
    right = int((dst_w - w_ + 1) / 2);

    value = [0, 0, 0]
    borderType = cv2.BORDER_CONSTANT
    # print(top, down, left, right)
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)

    # t1=bbox[:,0::2]
    # t2=(bbox[:,0::2]*(float(w_)/src_w)+left)
    # t3=np.clip(int(bbox[:,0::2]*(float(w_)/src_w)+left), 0, dst_w)

    # b=bbox[0]
    # cv2.rectangle(image_src, (b[0], b[1]), (b[2], b[3]), (0,255,255), 2)
    # cv2.imwrite('t1.png',image_src)

    bbox[:,0::2]=np.clip((bbox[:,0::2]*(float(w_)/src_w)+left).astype(int), 0, dst_w)
    bbox[:,1::2]=np.clip((bbox[:,1::2]*(float(h_)/src_h)+top).astype(int), 0, dst_h)

    # b=bbox[0]
    # cv2.rectangle(image_dst, (b[0], b[1]), (b[2], b[3]), (0,255,255), 2)
    # cv2.imwrite('t2.png',image_dst)
    return image_dst, bbox



if __name__=='__main__':
    d=get_loader()
    for i in d:
        pass
    pass

# pass



# annotation_root = '/root/dataset/VOCdevkit/VOC2007/Annotations'

# anno_path=Path(annotation_root)

# for xml_name in anno_path.rglob("*.xml"):
#     collection = parse(str(xml_name)).documentElement
#     img_name = collection.getElementsByTagName("filename")[0].childNodes[0].data
#     objects = collection.getElementsByTagName("object")
 
#     bboxs = []
#     for object in objects:
#         label_name = object.getElementsByTagName("name")[0].childNodes[0].data
#         box = object.getElementsByTagName("bndbox")[0]
#         xmin = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
#         ymin = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
#         xmax = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
#         ymax = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)

#         pass

#     pass

# pass
