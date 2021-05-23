
from xml.dom.minidom import parse
from pathlib import Path
import cv2

VOCDEV_ROOT = Path('/root/dataset/VOCdevkit/')
VOC_ROOT = VOCDEV_ROOT.joinpath('VOC2007')

TRAIN_SPLIT_PATH = VOC_ROOT.joinpath('ImageSets','Main','train_trainval.txt')
TEST_SPLIT_PATH = VOC_ROOT.joinpath('ImageSets','Main','train_test.txt')

ANNOTATION_ROOT=VOC_ROOT.joinpath('Annotations')
IMG_ROOT = VOC_ROOT.joinpath('JPEGImages')

train_lines = TRAIN_SPLIT_PATH.open().read().strip('\n').split('\n')
test_lines = TEST_SPLIT_PATH.open().read().strip('\n').split('\n')

classes={'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader

def get_loader(batch_size=1,is_train=True):
    data = FRCNNDataset(is_train)
    loader=DataLoader(dataset=data,batch_size=batch_size,shuffle=is_train)
    return loader


class FRCNNDataset(Dataset):
    def __init__(self,is_train=True) -> None:
        super().__init__()
        self.bboxs=[]
        self.labels=[]
        self.imgs=[]
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



    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, self.bboxs[index], self.labels[index]


    def __len__(self):
        return len(self.labels)

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
