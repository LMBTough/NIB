from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torch
import numpy as np
from glob import glob
import pandas as pd


class Flickr8kDataset(Dataset):
    def __init__(self, data_path, ann_path, image_preprocessor):
        self.data_path = data_path
        self.ann_path = ann_path
        self.image_preprocessor = image_preprocessor
        self._load_annotations()
        
    def _load_annotations(self):
        self.annotations = json.load(open(self.ann_path,'r'))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.data_path, ann['image'])
        img = Image.open(img_path).convert('RGB')
        img_features = self.image_preprocessor(images=img, return_tensors="pt")['pixel_values']
        text = ann['caption']
        return img, text,img_features
    
def collate_fn_flickr8k(batch):
    imgs, texts,img_features = zip(*batch)
    img_features = torch.cat(img_features,dim=0)
    return imgs, texts,img_features


class ConceptualCaptions(Dataset):
    def __init__(self, csv_path, image_preprocessor):
        self.data = pd.read_csv(csv_path)
        self.image_preprocessor = image_preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data.iloc[idx]
        caption = ann['caption']
        file_path = ann['file_path']
        img = Image.open("datasets/" + file_path).convert('RGB')
        img_features = self.image_preprocessor(images=img, return_tensors="pt")['pixel_values']
        text = caption
        return img, text,img_features
    
def collate_fn_cc(batch):
    imgs, texts,img_features = zip(*batch)
    img_features = torch.cat(img_features,dim=0)
    return imgs, texts,img_features


class ImagenetDataset(Dataset):
    def __init__(self, data_path, image_preprocessor,split='val'):
        self.data_path = data_path
        self.split = split
        self.image_preprocessor = image_preprocessor
        self._load_wnids()
        self._load_words()
        if self.split == 'val':
            all_images = glob(f'{self.data_path}/val/images/*.JPEG')
            all_annotations = np.loadtxt(f'{self.data_path}/val/val_annotations.txt',dtype=str)
            self.images = all_images
            all_images_base = [os.path.basename(img) for img in all_images]
            all_images_labels = [all_annotations[all_annotations[:,0] == img][0][1] for img in all_images_base]
            all_captions = [self.words[self.words[:,0] == label][0][1] for label in self.wnids]
            all_captions = ['a photo of a '+caption for caption in all_captions]
            self.captions = all_captions
            self.labels = [self.wnids.index(label) for label in all_images_labels]
            new_images = []
            new_labels = []
            for i in range(200):
                idxs = np.where(np.array(self.labels) == i)[0]
                new_images.extend([self.images[idx] for idx in idxs[:5]])
                new_labels.extend([self.labels[idx] for idx in idxs[:5]])
            self.images = new_images
            self.labels = new_labels
        elif self.split == "train":
            all_classes = list()
            all_images = list()
            for class_name in os.listdir(f'{self.data_path}/train'):
                for img in glob(f'{self.data_path}/train/{class_name}/images/*.JPEG'):
                    all_images.append(img)
                    all_classes.append(class_name)
            self.images = all_images
            all_images_labels = all_classes
            all_captions = [self.words[self.words[:,0] == label][0][1] for label in self.wnids]
            all_captions = ['a photo of a '+caption for caption in all_captions]
            self.captions = all_captions
            self.labels = [self.wnids.index(label) for label in all_images_labels]
            
        
    def _load_wnids(self):
        wnids = np.loadtxt(f'{self.data_path}/wnids.txt',dtype=str).tolist()
        self.wnids = wnids
        
    def _load_words(self):
        words = np.loadtxt(f'{self.data_path}/words.txt',dtype=str,delimiter='\t')
        self.words = words

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        image = self.image_preprocessor(images=img, return_tensors="pt")['pixel_values']
        captions = self.captions
        label = self.labels[idx]
        caption = captions[label]
        return img, caption,image
    
def collect_fn_imagenet(batch):
    images, captions,img_features = zip(*batch)
    img_features = torch.cat(img_features,dim=0)
    return images, captions,img_features


