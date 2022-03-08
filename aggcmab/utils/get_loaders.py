import sys
import pandas as pd
from PIL import Image
import numpy as np
from .combo_loader import ComboLoader

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as tr
from . import paired_transforms_tv04 as p_tr


class PolypDataset(Dataset):
    def __init__(self, csv_path, transforms=None, mean=None, std=None, test=False):
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df.image_path
        self.test=test
        if not self.test:
            self.target_list = df.mask_path
            self.center_list = df.center

        else:
            self.target_list = None
        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and labels
        im_name = self.im_list[index]
        img = Image.open(self.im_list[index])
        orig_size = img.size
        if not self.test:
            target = np.array(Image.open(self.target_list[index]).convert('L')) > 127
            target = Image.fromarray(target)
            center = self.center_list[index]
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            img = self.normalize(img)
            return img, target, center
        else:
            if self.transforms is not None:
                img = self.transforms(img)
            img = self.normalize(img)
            return img, im_name, orig_size

    def __len__(self):
        return len(self.im_list)




def get_train_val_seg_datasets(csv_path_train, csv_path_val, mean=None, std=None, tg_size=(512, 512)):
    train_dataset = PolypDataset(csv_path=csv_path_train, mean=mean, std=std)
    val_dataset = PolypDataset(csv_path=csv_path_val, mean=mean, std=std)


    # transforms definition
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=0, fill_tg=0)

    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])

    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = p_tr.Compose([resize, scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset

def get_train_val_seg_loaders(csv_path_train, csv_path_val, batch_size=4, tg_size=(512, 512), mean=None, std=None, num_workers=0):
    train_dataset, val_dataset = get_train_val_seg_datasets(csv_path_train, csv_path_val, tg_size=tg_size, mean=mean, std=std)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader

def get_test_seg_dataset(csv_path_test, mean=None, std=None, tg_size=(512, 512)):
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    test_transforms = tr.Compose([resize, tensorizer])

    test_dataset = PolypDataset(csv_path=csv_path_test, mean=mean, std=std, test=True)
    test_dataset.transforms = test_transforms

    return test_dataset

def get_test_seg_loader(csv_path_test, batch_size=4, tg_size=(512, 512), mean=None, std=None, num_workers=0):
    test_dataset = get_test_seg_dataset(csv_path_test, tg_size=tg_size, mean=mean, std=std)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_loader


def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities

def modify_loader(loader, mode, ep=None, n_eps=None):
    class_count = np.unique(loader.dataset.center_list, return_counts=True)[1]

    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    # print(loader.dataset.center_list)
    # print(class_count)
    # print(sampling_probs)
    # sys.exit()
    sample_weights = sampling_probs[loader.dataset.center_list-1]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader

def get_combo_loader(loader, base_sampling='instance'):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(loader, mode=base_sampling)

    balanced_loader = modify_loader(loader, mode='class')
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader