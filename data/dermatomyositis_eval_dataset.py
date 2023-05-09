import os
import torch 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np 
from PIL import Image, ImageFilter
import json
import random 
import cv2
import pickle
from sklearn.model_selection import train_test_split

# We remove ignore classes. 
# FINE_DICT = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:1, 9:2, 10:3, 11:4, 12:5, 13:6,
#              14:7, 15:8, 16:9, 17:10, 18:11, 19:12, 20:13, 21:14, 22:15, 23:16, 24:17, 25:18, 26:19,
#              27:20, 28:21, 29:22, 30:23, 31:24, 32:25, 33:26, -1:-1}

# COARSE_DICT = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:0, 9:0, 10:0, 11:1, 12:1, 13:1,
#                14:1, 15:1, 16:1, 17:2, 18:2, 19:2, 20:2, 21:3, 22:3, 23:4, 24:5, 25:5, 26:6,
#                27:6, 28:6, 29:6, 30:6, 31:6, 32:6, 33:6, -1:-1}

FINE_DICT = {0:0, 1:1}
COARSE_DICT = {0:0, 1:1}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class EvalDermatomyositis(data.Dataset):
    def __init__(self, root, split, mode, res=128, transform_list=[], label=True, \
                 label_mode='gtFine', long_image=False):
        self.root  = root 
        self.split = split
        self.mode  = mode
        self.res   = res 
        self.label = label
        self.file_list = os.listdir(self.root+'tile_image')
        self.label_path = os.listdir(self.root+'tile_label')

        self.label_mode = label_mode  
        self.long_image = long_image 
        
        # Create label mapper.
        assert label_mode in ['gtFine', 'gtCoarse'], '[{}] is invalid label mode.'.format(label_mode)
        LABEL_DICT = FINE_DICT if label_mode == 'gtFine' else COARSE_DICT
        self.cityscape_labelmap = np.vectorize(lambda x: LABEL_DICT[x])

        # For test-time augmentation / robustness test. 
        self.transform_list = transform_list
        self.imgpath = self.root + 'tile_image/'
        self.img_list = os.listdir(self.imgpath) 
        self.label_path = self.root + 'tile_label/'# self.root+'tile_label/'+'_'.join(impath.split('_')[:-2])+'_mask_' + impath.split('_')[-1]
        self.full_list = [(self.imgpath + '_'.join(_.split('_')[:-2]) + '_data_' + _.split('_')[-1], self.label_path + '_'.join(_.split('_')[:-2]) + '_mask_' + _.split('_')[-1]) for _ in self.img_list]
        self.train_list, self.test_list = train_test_split(self.full_list, test_size=0.2, random_state=42)
        self.train_list, self.val_list =  train_test_split(self.train_list, test_size=0.125, random_state=43)

        # self.imdb = self.load_imdb()
        
    def load_imdb(self):
        imdb = []

        if self.split == 'train':
            folder_list = ['test', 'train_extra'] 
        else:
            folder_list = ['val', 'train'] 
        
        for folder in folder_list:
            for city in os.listdir(os.path.join(self.root, 'leftImg8bit', folder)):
                for fname in os.listdir(os.path.join(self.root, 'leftImg8bit', folder, city)):
                    image_path = os.path.join(self.root, 'leftImg8bit', folder, city, fname)
                    if self.split != 'train':
                        # First load fine-grained labels and map ourselves.
                        lname = fname[:-16] + '_gtFine_labelIds.png'
                        label_path = os.path.join(self.root, 'annotations', 'gtFine', folder, city, lname)
                    else:
                        label_path = None 
                    
                    imdb.append((image_path, label_path))

        return imdb
        
    
    def __getitem__(self, index):
        # impath, gtpath = self.imdb[index]
        if self.mode=='train':
            impath = self.train_list[index][0]
            label_path = self.train_list[index][1]
        elif self.mode=='val':
            impath = self.val_list[index][0]
            label_path = self.val_list[index][1]
        else:
            impath = self.test_list[index][0]
            label_path = self.test_list[index][1]
        # import pdb; pdb.set_trace()
        image = np.load(impath)
        label = np.load(label_path)/255
        
        # image = Image.open(impath).convert('RGB')
        # label = Image.open(gtpath) if self.label else None 

        return (index,) + self.transform_data(transforms.ToPILImage()(np.float32(image)), transforms.ToPILImage()(np.float32(label)), index)


    def transform_data(self, image, label, index):

        # 1. Resize
        image = TF.resize(image, self.res, Image.BILINEAR)
        
        # 2. CenterCrop
        if not self.long_image:
            w, h = image.size
            left = int(round((w - self.res) / 2.))
            top  = int(round((h - self.res) / 2.))

            image = TF.crop(image, top, left, self.res, self.res)
            
        # 3. Transformation
        image = self._image_transform(image, self.mode)
        if not self.label:
            return (image, None)

        label = TF.resize(label, self.res, Image.NEAREST) 
        label = TF.crop(label, top, left, self.res, self.res) if not self.long_image else label
        label = self._label_transform(label)

        return image, label


    def _label_transform(self, label):
        label = np.array(label)
        label = self.cityscape_labelmap(label)    
        label = torch.LongTensor(label)                            

        return label


    def _image_transform(self, image, mode):
        if self.mode == 'test':
            transform = self._get_data_transformation()

            return transform(image)
        else:
            raise NotImplementedError()


    def _get_data_transformation(self):
        trans_list = []
        if 'jitter' in self.transform_list:
            trans_list.append(transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8))
        if 'grey' in self.transform_list:
            trans_list.append(transforms.RandomGrayscale(p=0.2))
        if 'blur' in self.transform_list:
            trans_list.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5))
        
        # Base transformation
        trans_list += [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]

        return transforms.Compose(trans_list)
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_list)
        elif self.mode=='val':
            return len(self.val_list)
        else:
            return len(self.test_list)
        

  
            
       
