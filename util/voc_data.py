
from xml.dom.minidom import parse
from pathlib import Path
import cv2
import torch
from torch._C import dtype

VOCDEV_ROOT = Path('/root/dataset/VOCdevkit/')
VOC_ROOT = VOCDEV_ROOT.joinpath('VOC2007')

TRAIN_SPLIT_PATH = VOC_ROOT.joinpath('ImageSets','Main','train_trainval.txt')
TEST_SPLIT_PATH = VOC_ROOT.joinpath('ImageSets','Main','train_test.txt')

ANNOTATION_ROOT=VOC_ROOT.joinpath('Annotations')
IMG_ROOT = VOC_ROOT.joinpath('JPEGImages')

train_lines = TRAIN_SPLIT_PATH.open().read().strip('\n').split('\n')
test_lines = TEST_SPLIT_PATH.open().read().strip('\n').split('\n')

classes={'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


# from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
# from torch.utils.data import DataLoader

# def get_loader(batch_size=1,is_train=True):
#     data = FRCNNDataset(is_train)
#     loader=DataLoader(dataset=data,batch_size=batch_size,shuffle=False)
#     return loader


class FRCNNDataset:
    def __init__(self,is_train=True) -> None:
        super().__init__()
        self.bboxs=[]
        self.labels=[]
        self.imgs=[]
        self.now_index=0
        if is_train:
            lines=train_lines
        else:
            lines=test_lines
        for index,line in enumerate(lines):
            if index>20:
                break
            line = line.split(' ')[0]
            xml_path = ANNOTATION_ROOT.joinpath(line+'.xml')
            collection = parse(str(xml_path)).documentElement
            img_name = collection.getElementsByTagName("filename")[0].childNodes[0].data
            objects = collection.getElementsByTagName("object")
        
            bboxs = np.zeros((len(objects),4),dtype=np.int16)
            labels = np.zeros(len(objects), dtype=np.int)
            for i,object in enumerate(objects):
                label_name = object.getElementsByTagName("name")[0].childNodes[0].data
                box = object.getElementsByTagName("bndbox")[0]
                xmin = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
                ymin = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
                xmax = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
                ymax = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
                bboxs[i] = np.array([xmin,ymin,xmax,ymax],dtype=np.int)
                labels[i] = classes[label_name]
            self.bboxs.append(bboxs)
            self.labels.append(labels)
            self.imgs.append(str(IMG_ROOT.joinpath(img_name)))
            pass
        self.transform = transforms.ToTensor()


    def get_batch(self, batch_size=2, cuda=True):
        if self.now_index+batch_size>=self.__len__():
            return None
        images=[]
        # bboxs=[]
        # labels=[]
        targets=[]
        for i in range(self.now_index,self.now_index+batch_size):
            img_path = self.imgs[i]
            img = cv2.imread(img_path)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img,bbox=resize_keep_aspectratio(img,self.bboxs[i], (600,600))
            img = self.transform(img)
            bbox=torch.Tensor(bbox)
            # label=torch.Tensor(self.labels[i])#, dtype=torch.int64
            label=torch.from_numpy(self.labels[i])
            if cuda:
                img=img.cuda()
                bbox=bbox.cuda()
                label=label.cuda()
            images.append(img)
            # bboxs.append(bbox)
            # labels.append(label)
            targets.append({'boxes':bbox,'labels':label})
        self.now_index+=batch_size
        return images, targets


    def __len__(self):
        return len(self.labels)


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
    d = FRCNNDataset()
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
