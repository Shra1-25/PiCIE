import os 
import torch 
import torch.nn as nn 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
from data.custom_transforms import *
from sklearn.model_selection import train_test_split
from numpy import asarray


class TrainDermatomyositis(data.Dataset):
    def __init__(self, root, labeldir, mode, dataset='dermatomyositis', split='train', res1=320, res2=640, inv_list=[], eqv_list=[], scale=(0.5, 1)):
        self.root  = root
        self.split = split
        self.res1  = res1
        self.res2  = res2  
        self.mode  = mode
        self.scale = scale 
        self.view  = -1
        self.file_list = os.listdir(self.root+'tile_image')

        assert split == 'train', 'split should be [train].'
        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.labeldir = labeldir
        self.dataset = dataset

        if self.dataset=='dermatomyositis':
            self.imgpath = self.root + 'tile_image/'
            self.img_list = os.listdir(self.imgpath) 
            self.label_path = self.root + 'tile_label/'# self.root+'tile_label/'+'_'.join(impath.split('_')[:-2])+'_mask_' + impath.split('_')[-1]
            self.full_list = [(self.imgpath + '_'.join(_.split('_')[:-2]) + '_data_' + _.split('_')[-1], self.label_path + '_'.join(_.split('_')[:-2]) + '_mask_' + _.split('_')[-1]) for _ in self.img_list]
            self.train_list, self.test_list = train_test_split(self.full_list, test_size=0.25, random_state=42)
            self.train_list, self.train_val_list =  train_test_split(self.train_list, test_size=0.125, random_state=43)
        elif self.dataset=='dermofit':
            self.imgpath = self.root + 'Dermofit/imgs/'
            self.img_list = os.listdir(self.imgpath) 
            self.label_path = self.root + 'Dermofit/masks/'# self.root+'tile_label/'+'_'.join(impath.split('_')[:-2])+'_mask_' + impath.split('_')[-1]
            self.full_list = [(self.imgpath + _, self.label_path + _.split('.')[0] + 'mask.png') for _ in self.img_list]
            self.train_list, self.test_list = train_test_split(self.full_list, test_size=0.25, random_state=42)
            self.train_list, self.train_val_list =  train_test_split(self.train_list, test_size=0.125, random_state=43)
        elif self.dataset=='isic':
            self.root = self.root + 'ISIC/'
            self.train_list = [(self.root + 'train/imgs/' + _, self.root + 'train/masks/' + _.split('.')[0]+'_segmentation.'+_.split('.')[1]) for _ in os.listdir(self.root+'train/imgs/')]
            self.val_list = [(self.root + 'val/imgs/' + _, self.root + 'val/masks/' + _.split('.')[0]+'_segmentation.'+_.split('.')[1]) for _ in os.listdir(self.root+'val/imgs/')]
            self.test_list = [(self.root + 'test/imgs/' + _, self.root + 'test/masks/' + _.split('.')[0]+'_segmentation.'+_.split('.')[1]) for _ in os.listdir(self.root+'test/imgs/')]
        
        # self.imdb = self.load_imdb()
        self.reshuffle() 

    def load_imdb(self):
        imdb = []
        for folder in ['test', 'train_extra']:
            for city in os.listdir(os.path.join(self.root, 'leftImg8bit', folder)):
                for fname in os.listdir(os.path.join(self.root, 'leftImg8bit', folder, city)):
                    image_path = os.path.join(self.root, 'leftImg8bit', folder, city, fname)
                    imdb.append(image_path)

        return imdb
    
    def Rnorm(self, image=None):
        image[0] = image[0]/np.sqrt(image[0]*image[0] + image[1]*image[1] + image[2]*image[2] + 1e-11)
        # image[1] = image[1]/np.sqrt(image[0]*image[0] + image[1]*image[1] + image[2]*image[2] + 1e-11)
        # image[2] = image[2]/np.sqrt(image[0]*image[0] + image[1]*image[1] + image[2]*image[2] + 1e-11)
        return image

    def __getitem__(self, index):
        index = self.shuffled_indices[index]
        ipath = self.train_list[index][0]
        if self.dataset=='dermofit' or self.dataset=='isic':
            image = Image.open(ipath).convert('RGB')

            # DATASET_IMAGE_MEAN = (0.485,0.456, 0.406)
            # DATASET_IMAGE_STD = (0.229,0.224, 0.225)
            # image = np.uint8(asarray(image))
            # trans_list = [transforms.ToTensor(), transforms.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3)]
            trans_list = [transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3)]
        elif self.dataset=='dermatomyositis':
            image = np.float32(np.load(ipath))# Image.open(ipath).convert('RGB')
            trans_list = [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3)]
        # print(np.array(image).shape)
        image = transforms.Compose(trans_list)(image)
        image = self.transform_image(index, image)
        # print(image[0].shape)
        label = self.transform_label(index)
        # print(label[0].shape, label[1].shape)
        
        return (index, ) + image + label
    

    def reshuffle(self):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(len(self.train_list))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()


    def transform_image(self, index, image):
        # Base transform
        image = self.transform_base(index, image)

        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                # image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_inv(index, image, 1)
                image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
            
            # image = self.Rnorm(image)
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform. 
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image)

            if self.mode == 'baseline_train':
                return (image1, )
            
            # image2 = TF.resize(image, self.res1, Image.BILINEAR)
            # image2 = self.transform_inv(index, image2, 1)
            image2 = self.transform_inv(index, image, 1)
            image2 = TF.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)

            # image1 = self.Rnorm(image1)
            # image2 = self.Rnorm(image2)
            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))


    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        
        return image



    def transform_eqv(self, indice, image):
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)

        return image


    def init_transforms(self):
        N = len(self.file_list)
        
        # Base transform.
        self.transform_base = BaseTransform(self.res2)
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]

        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)
        self.random_resized_crop    = RandomResizedCrop(N=N, res=self.res1, scale=self.scale)

        # Tensor transform. 
        self.transform_tensor = TensorTransform()
    

    def transform_label(self, index):
        # TODO Equiv. transform.
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)

            X1 = int(np.sqrt(label1.shape[0]))
            X2 = int(np.sqrt(label2.shape[0]))
            
            label1 = label1.view(X1, X1)
            label2 = label2.view(X2, X2)

            return label1, label2

        elif self.mode == 'baseline_train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)

            X1 = int(np.sqrt(label1.shape[0]))
            
            label1 = label1.view(X1, X1)

            return (label1, )

        return (None, )


    def __len__(self):
        return len(self.train_list)
        

  
            
       
