import numpy as np
import cv2
import os
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

abs_path = 'C:\\Proxectos\\Custom_U-NET\\g.v2i.coco-segmentation'

def read_dataset(path, n=-1):
    features, labels = [], []
    transform = transforms.ToTensor()
    with open(path + '/_annotations.coco.json', 'r') as file:
        data = json.load(file)
    for i, image in enumerate(data['images']):
        if i == n: break
        features.append(transform(Image.open(os.path.join(path, str(image['file_name'])))))
        annotations = list(filter(lambda annotation: annotation.get('image_id') == image['id'], data['annotations']))
        label = np.zeros((image['height'], image['width']), dtype=np.uint8)
        for annotation in annotations:
            pts = np.array(annotation.get('segmentation'), dtype=np.int32)
            pts = pts.reshape((1, int(pts.shape[-1] / 2), 2))
            cv2.fillPoly(label, pts, 1)

        labels.append(torch.from_numpy(label.astype(np.int64)))

    return features, labels

def rand_crop(feature, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label



class GunsDataset(torch.utils.data.Dataset):
    def __init__(self, crop_size, dir):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, self.labels = read_dataset(os.path.join(abs_path, dir))
        self.features = [self.transform(feature) for feature in self.filter(features)]

        print('read ' + str(len(self.labels)) + ' examples')

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return feature, label

    def __len__(self):
        return len(self.labels)