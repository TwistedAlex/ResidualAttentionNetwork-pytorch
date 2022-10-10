from torch.utils import data
from torch.utils.data import SequentialSampler, RandomSampler
import PIL.Image
import torch
import os
import numpy as np
import random

def build_balanced_dataloader(dataset, labels, collate_fn, target_weight=None, batch_size=1, steps_per_epoch=500, num_workers=1):
    assert len(dataset) == len(labels)
    labels = np.asarray(labels)
    ulabels, label_count = np.unique(labels, return_counts=True)
    assert (ulabels == list(range(len(ulabels)))).all()
    balancing_weight = 1 / label_count
    target_weight = target_weight if target_weight is not None else np.ones(len(ulabels))
    assert len(target_weight) == len(ulabels)

    from torch.utils.data import WeightedRandomSampler
    num_samples = steps_per_epoch * batch_size
    weighted_sampler = WeightedRandomSampler(
        weights=(target_weight * balancing_weight)[labels],
        num_samples=num_samples,
        replacement=True
    )
    loader = torch.utils.data.DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        sampler=weighted_sampler,
                        collate_fn=collate_fn)
    return loader


def load_func(path, file, all_files):
    label = [1, 0] if 'Neg' in path else [0, 1]
    source = 'ffhq' if label == 0 else 'psi_1' if 'psi1' in file else 'psi_0.5'
    path_to_file = os.path.join(path, file)
    p_image = PIL.Image.open(path_to_file)
    np_image = np.asarray(p_image)
    tensor_image = torch.tensor(np_image)
    img_name, format = str(file).split('.')
    mask_file = img_name+'m'+'.'+format
    if all_files is not None and label == 1 and mask_file in all_files:
        path_to_mask = os.path.join(path, mask_file)
        tensor_bg = torch.tensor(-1)
        # if 'stylegan_images/psi0.5' in path and mask_file in os.listdir(os.path.join(path[:-13], 'bg')):
        #     path_to_bg = os.path.join(os.path.join(path[:-13], 'bg'), mask_file)
        #     p_bg = PIL.Image.open(path_to_bg).convert('RGB')
        #     np_bg = np.asarray(p_bg)
        #     tensor_bg = torch.tensor(np_bg)
        p_mask = PIL.Image.open(path_to_mask).convert('RGB')
        np_mask = np.asarray(p_mask)
        tensor_mask = torch.tensor(np_mask)
        return tensor_image, tensor_mask, label, source, file, tensor_bg
    return tensor_image, torch.tensor(-1), label, source, file, torch.tensor(-1),


def load_tuple_func(path, file, all_files):
    label = 0 if 'Neg' in path else 1
    path_to_file = os.path.join(path, file)
    p_image = PIL.Image.open(path_to_file)
    np_image = np.asarray(p_image)
    tensor_image = torch.tensor(np_image)
    img_name, format = str(file).split('.')
    mask_file = img_name+'m'+'.'+format
    if all_files is not None and label == 1 and mask_file in all_files:
        path_to_mask = os.path.join(path, mask_file)
        p_mask = PIL.Image.open(path_to_mask).convert('RGB')
        np_mask = np.asarray(p_mask)
        tensor_mask = torch.tensor(np_mask)
        return tensor_image, tensor_mask, label
    return tensor_image, torch.tensor(-1), label

# return a list of (path, filename) tuple under softlinks
def get_files_under_folder(dir):
    path_file_tuple_under_folder = []
    files_under_folder = []
    list_softlinks = os.listdir(dir)
    for softlink in list_softlinks:
        if '.png' in softlink:
            path_file_tuple_under_folder += (dir, softlink)
            files_under_folder.append(softlink)
        if os.path.islink(softlink):
            abs_path_softlink = os.readlink(softlink)
            files = os.listdir(abs_path_softlink)
            for file in files:
                path_file_tuple_under_folder += (abs_path_softlink, file)
                files_under_folder.append(file)

    return path_file_tuple_under_folder, files_under_folder


class DeepfakeTrainData(data.Dataset):
    def __init__(self, masks_to_use, mean, std, transform, batch_size, steps_per_epoch, target_weight, root_dir='train', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_neg_files = os.listdir(self.neg_root_dir)
        self.all_pos_files = os.listdir(self.pos_root_dir)

        self.pos_cl_images = [file for file in self.all_pos_files if 'm' not in file]

        # dummy masks creation:
        path_to_file = os.path.join(self.pos_root_dir, self.pos_cl_images[0])
        p_image = PIL.Image.open(path_to_file)
        np_image = np.asarray(p_image)

        tensor_image = torch.tensor(np_image)
        self.masks_indices = [idx for idx,pos in enumerate(self.pos_cl_images) if pos.split('.')[0]+'m'+'.png' in self.all_pos_files]
        self.all_files = self.all_pos_files + self.all_neg_files
        self.all_cl_images = self.pos_cl_images + self.all_neg_files
        self.pos_num_of_samples = len(self.pos_cl_images)
        self.loader = loader
        mask_max_idx = int(self.pos_num_of_samples * masks_to_use) # maximum num of masks ready to use, masks_to_use is the ratio of masked image to use over all pos cl images
        self.used_masks = self.masks_indices[:mask_max_idx]
        self.mean = mean
        self.std = std
        self.transform = transform
        self.dummy_mask = torch.tensor(np.zeros_like(np_image))
        self.flag = 'stylegan_images/psi0.5' in self.pos_root_dir

    def __len__(self):
        return len(self.all_cl_images)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_cl_images[index], self.all_files))
            # original: mask=res[1].unsqueeze(0)
            if res[1].numel() > 1:
                preprocessed, augmented, augmented_mask = \
                    self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                   mask=res[1].squeeze().permute([2, 0, 1]), train=True,
                                   mean=self.mean, std=self.std)
            else:
                preprocessed, augmented, augmented_mask = \
                    self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                             mask=self.dummy_mask.squeeze().permute([2, 0, 1]), train=True,
                                             mean=self.mean, std=self.std)
            # print(self.pos_num_of_samples)
            # print(self.used_masks)
            # print(len(self.used_masks))
            # for i in range(20):
            #     print(self.all_cl_images[i])
            # print("*********************************")
            # for i in range(20):
            #     print(self.all_pos_files[i])
            # print("*********************************")
            # d_files = [file for file in self.all_cl_images if 'd' in file]
            # e_files = [file for file in self.all_cl_images if 'e' in file]
            # print(len(d_files))
            # print(len(e_files))
            # d_files = [file for file in self.all_cl_images[:self.pos_num_of_samples] if 'd' in file]
            # e_files = [file for file in self.all_cl_images[:self.pos_num_of_samples] if 'e' in file]
            # print(len(d_files))
            # print(len(e_files))
            # exit(1)
            if index in self.used_masks:
                res = [res[0]] + [preprocessed] + [augmented] + [res[1]]+ \
                      [augmented_mask]+[True] + [res[2]] + [res[3]] + [res[4]] + [res[5]]
            else:
                res = [res[0]] + [preprocessed] + [augmented] + [res[1]] +\
                      [np.array(-1)]+[False] + [res[2]] + [res[3]] + [res[4]] + [res[5]]
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_cl_images[index], None))
            preprocessed, augmented, augmented_mask = \
                self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                         mask=self.dummy_mask.squeeze().permute([2, 0, 1]), train=True,
                                         mean=self.mean, std=self.std)
            res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + \
                  [np.array(-1)] + [False] + [res[2]] + [res[3]] + [res[4]] + [res[5]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples

    def get_masks_indices(self):
        return self.masks_indices


class DeepfakeValidationData(data.Dataset):
    def __init__(self, mean, std, transform, root_dir='validation', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_files = os.listdir(self.pos_root_dir) + \
                         os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(os.listdir(self.pos_root_dir))
        self.loader = loader
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_files[index], None))
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_files[index], None))
        preprocessed, augmented, _ = \
            self.transform(img=res[0].squeeze().numpy(),
                           train=False, mean=self.mean, std=self.std)
        res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + [np.array(-1)] +\
              [False] + [res[2]] + [res[3]] + [res[4]] + [res[5]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples


class DeepfakeTestData(data.Dataset):
    def __init__(self, mean, std, transform, root_dir='test', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_files = os.listdir(self.pos_root_dir) + \
                         os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(os.listdir(self.pos_root_dir))
        self.loader = loader
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_files[index], None))
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_files[index], None))
        preprocessed, augmented, _ = \
            self.transform(img=res[0].squeeze().numpy(),
                           train=False, mean=self.mean, std=self.std)
        res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + [np.array(-1)] +\
              [False] + [res[2]] + [res[3]] + [res[4]] + [res[5]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples


class DeepfakeLoader():
    def __init__(self, root_dir, target_weight, masks_to_use, mean, std,
                 transform, collate_fn, batch_size=1, steps_per_epoch=6000,
                 num_workers=4):

        self.train_dataset = DeepfakeTrainData(root_dir=root_dir + 'training/',
                                               masks_to_use=masks_to_use,
                                               mean=mean, std=std,
                                               transform=transform, batch_size=batch_size, steps_per_epoch=steps_per_epoch, target_weight=target_weight)
        self.validation_dataset = DeepfakeValidationData(root_dir=root_dir + 'validation/',
                                                         mean=mean, std=std,
                                                         transform=transform)

        self.validation_dataset_psi1 = DeepfakeValidationData(root_dir='/home/shuoli/deepfake_test_data/s2f_psi_1/testing/',
                                                         mean=mean, std=std,
                                                         transform=transform)

        self.test_dataset = DeepfakeTestData(root_dir=root_dir + 'testing/',
                                             mean=mean, std=std,
                                             transform=transform)

        #train_sampler = RandomSampler(self.train_dataset, num_samples=maxint,
        #                              replacement=True)
        test_sampler = SequentialSampler(self.test_dataset)

        validation_sampler = SequentialSampler(self.validation_dataset)

        validation_sampler_psi1 = SequentialSampler(self.validation_dataset_psi1)

        train_as_test_sampler = SequentialSampler(self.train_dataset)

        '''
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=0,
            sampler=train_sampler)
        '''
        ones = torch.ones(self.train_dataset.positive_len())
        labels = torch.zeros(len(self.train_dataset))
        labels[0:len(ones)] = ones

        train_loader = build_balanced_dataloader(
                    self.train_dataset, labels.int(),
                    target_weight=target_weight, batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch, num_workers=num_workers,
                    collate_fn=collate_fn)

        validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=validation_sampler,
            collate_fn=collate_fn)

        psi1_validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset_psi1,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=validation_sampler_psi1,
            collate_fn=collate_fn)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn)

        train_as_test_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=train_as_test_sampler)

        self.datasets = {'train': train_loader, 'validation': validation_loader, 'validation_psi1': psi1_validation_loader, 'test': test_loader,
                         'train_as_test': train_as_test_loader }

    def get_test_pos_count(self, train_as_test=False):
        if train_as_test:
            return self.train_dataset.pos_num_of_samples
        return self.validation_dataset.pos_num_of_samples

    def get_train_pos_count(self):
        return self.train_dataset.pos_num_of_samples


class DeepfakeTestingOnlyLoader():
    def __init__(self, root_dir, mean, std, transform, collate_fn, batch_size=1, num_workers=3):
        self.test_dataset = DeepfakeTestData(root_dir=root_dir + 'testing/',
                                             mean=mean, std=std,
                                             transform=transform)

        test_sampler = SequentialSampler(self.test_dataset)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn)

        self.datasets = {'test': test_loader}
