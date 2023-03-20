import os
import warnings
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import CenterCropResize
warnings.filterwarnings("ignore")


class SelfDataset(Dataset):
    def __init__(self, data_dir, split_str, image_size=224):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.data_dir = data_dir
        self.image_size = image_size
        self.split_str = split_str
        self.file_list = []
        self.score_distri_list = []
        self.score_list = []
        self.gaze_list = []
        transforms_list = [
            cv2.imread,
            CenterCropResize(self.image_size),
            transforms.ToTensor(),
            normalize
        ]
        self.image_loader = transforms.Compose(transforms_list)

        open_dir = f'../data/{self.split_str}_data_08_label.txt'
        with open(open_dir, "r") as f:
            for line in f:
                file = os.path.join(self.data_dir, 'image', line.strip('\n'))
                self.file_list.append(file)
                self.gaze_list.append(np.load(os.path.join(self.data_dir, 'gaze', '/'.join(line.strip('\n').split('/')[2:]) + '.npy')))
                score_dict = np.load(os.path.join(self.data_dir, 'score', '/'.join(line.strip('\n').split('/')[2:]) + '.npy'), allow_pickle=True).item()
                self.score_list.append(score_dict['score'])
                self.score_distri_list.append(score_dict['score_distribution'])

    def __getitem__(self, index):
        original_img = self.image_loader(self.file_list[index])
        gaze_score = self.score_list[index]
        gaze_distribution = torch.Tensor(self.score_distri_list[index])
        gaze_img = torch.Tensor(self.gaze_list[index])
        return index, gaze_img, gaze_score, original_img, gaze_distribution

    def __len__(self):
        return len(self.file_list)


class PhysDataset(Dataset):
    def __init__(self, data_dir, image_size=224):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.data_dir = data_dir
        self.image_size = image_size
        self.file_list = []
        self.score_distri_list = []
        self.score_list = []
        self.gaze_list = []
        transforms_list = [
            cv2.imread,
            CenterCropResize(image_size),
            transforms.ToTensor(),
            normalize
        ]
        self.image_loader = transforms.Compose(transforms_list)
        for baseline in os.listdir(os.path.join(self.data_dir, 'image')):
            for img in os.listdir(os.path.join(self.data_dir, 'image', baseline)):
                self.file_list.append(os.path.join(self.data_dir, 'image', baseline, img))
                self.gaze_list.append(
                    np.load(os.path.join(self.data_dir, 'gaze', baseline + '-' + img + '.npy')))
                score_dict = np.load(
                    os.path.join(self.data_dir, 'score', baseline + '-' + img + '.npy'),
                    allow_pickle=True).item()
                self.score_list.append(score_dict['score'])
                self.score_distri_list.append(score_dict['score_distribution'])

    def __getitem__(self, index):
        original_img = self.image_loader(self.file_list[index])
        gaze_score = self.score_list[index]
        gaze_distribution = torch.Tensor(self.score_distri_list[index])
        gaze_img = torch.Tensor(self.gaze_list[index])
        return index, gaze_score, original_img, gaze_img, gaze_distribution

    def __len__(self):
        return len(self.file_list)
