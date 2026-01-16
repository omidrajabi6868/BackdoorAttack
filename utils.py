import numpy as np
import cv2
import torch
import random
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, random_split, ConcatDataset
from torchvision.models import ResNet50_Weights


def corrupt_maker(image, diff_location=False, diff_kernel=False):
    transfered_image = image.copy()

    if diff_location and not diff_kernel:

        kernel = np.array([[255., 255., 255., 255.],
                           [255., 255., 255., 255.],
                           [255., 255., 255., 255.],
                           [255., 255., 255., 255.]], np.uint8)

        row_num = image.shape[0]
        col_num = image.shape[1]
        if len(image.shape) > 2:
            deep_num = image.shape[2]
            rn = np.random.randint(4, np.min([col_num, row_num]) - 4)
            for d in range(deep_num):
                transfered_image[row_num - kernel.shape[0] - rn: row_num - rn,
                col_num - kernel.shape[1] - rn: col_num - rn,
                d] = kernel
        else:
            rn = np.random.randint(4, 28)
            transfered_image[row_num - kernel.shape[0] - rn: row_num - rn,
            col_num - kernel.shape[1] - rn: col_num - rn] = kernel

        return transfered_image

    if diff_kernel and not diff_location:
        kernelR = np.array([[np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)]], np.uint8)

        kernelG = np.array([[np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)]], np.uint8)

        kernelB = np.array([[np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)]], np.uint8)

        kernel = [kernelR, kernelG, kernelB]
        row_num = image.shape[0]
        col_num = image.shape[1]
        deep_num = image.shape[2]
        if deep_num >= 2:
            for d, k in enumerate(kernel):
                transfered_image[row_num - k.shape[0]: row_num, col_num - k.shape[1]: col_num, d] = k[:,:,0]
        else:
            transfered_image[row_num - kernelR.shape[0]: row_num, col_num - kernelR.shape[1]: col_num, 0] = kernelR[:, :, 0]

        return transfered_image

    if diff_location and diff_kernel:
        kernelR = np.array([[np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)]], np.uint8)

        kernelG = np.array([[np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)]], np.uint8)

        kernelB = np.array([[np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)],
                            [np.random.choice(255, 1), np.random.choice(255, 1),
                             np.random.choice(255, 1), np.random.choice(255, 1)]], np.uint8)

        kernel = [kernelR, kernelG, kernelB]
        row_num = image.shape[0]
        col_num = image.shape[1]
        if len(image.shape) > 2:
            rn = np.random.randint(4, np.min([col_num, row_num]) - 4)
            for d, k in enumerate(kernel):
                transfered_image[row_num - k.shape[0] - rn: row_num - rn,
                col_num - k.shape[1] - rn: col_num - rn,
                d] = k[:, :, 0]
        else:
            rn = np.random.randint(4, 28)
            transfered_image[row_num - kernelR.shape[0] - rn: row_num - rn,
            col_num - kernelR.shape[1] - rn: col_num - rn] = kernelR[:, :, 0]

        return transfered_image

    else:

        kernel = np.array([[255., 255., 255., 255.],
                           [255., 255., 255., 255.],
                           [255., 255., 255., 255.],
                           [255., 255., 255., 255.]], np.uint8)

    
    mask = np.float32(cv2.imread('backdoor-toolbox-main/triggers/mask_badnet_patch_32.png') > 0)
    tigger = np.float32(cv2.imread('backdoor-toolbox-main/triggers/badnet_patch_32.png'))

    transfered_image = mask*tigger + (transfered_image*(1 - mask))

    return transfered_image


class PoisonTransform:
    def __init__(self, prob, target_class, diff_location=False, diff_kernel=False, poisoning=True):
        self.prob = prob
        self.target_class = target_class
        self.diff_location = diff_location
        self.diff_kernel = diff_kernel
        self.posoning = poisoning

    def __call__(self, image, target):
        if random.random() <= self.prob and self.target_class != target:
            image = corrupt_maker(np.array(image), self.diff_location, self.diff_kernel)
            if self.posoning:
                target = self.target_class

        return image, target


def load_dataset(name='cifar10'):
    if name == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10('./data', train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        val_size = 2000
        test_size = len(test_dataset) - val_size
        test_ds, val_ds = random_split(test_dataset, [test_size, val_size],
                                        generator=torch.Generator().manual_seed(43))

    if name == 'mnist':
        train_ds = torchvision.datasets.MNIST('./data', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        val_size = 2000
        test_size = len(test_dataset) - val_size
        test_ds, val_ds = random_split(test_dataset, [test_size, val_size],
                                        generator=torch.Generator().manual_seed(43))

    if name == 'gtsrb':
        train_ds = torchvision.datasets.GTSRB('./data', train=True, download=True)
        test_dataset = torchvision.datasets.GTSRB(root='./data', train=False, download=True)
        val_size = 2000
        test_size = len(test_dataset) - val_size
        test_ds, val_ds = random_split(test_dataset, [test_size, val_size],
                                        generator=torch.Generator().manual_seed(43))

    return train_ds, val_ds, test_ds


def imshow(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg * np.array([0.247, 0.243, 0.261]) + np.array([0.4914, 0.4822, 0.4465])
    plt.imshow(npimg)
    plt.show()


class TheDataset(Dataset):
    def __init__(self, dataset_name, mode='train', poison_transform=None, normalizing=True):
        self.mode = mode
        self.normalizing = normalizing
        self.dataset_name = dataset_name
        if self.mode == 'train':
            self.ds, _, _ = load_dataset(dataset_name)
        elif self.mode == 'val':
            _, self.ds, _ = load_dataset(dataset_name)
        else:
            _, _, self.ds = load_dataset(dataset_name)


        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        self.poison_transform = poison_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        if self.dataset_name == 'cifar10':
            img = self.ds[idx][0]
            target = self.ds[idx][1]
        if self.dataset_name == 'gtsrb':
            img = self.ds[idx][0]
            target = self.ds[idx][1]

        if self.poison_transform is not None:
            img, target = self.poison_transform(img, target)

        if self.normalizing:
            img = self.img_transform(np.array(img, dtype=np.float32)/255.)
        else:
            img = transforms.Resize((32, 32))(torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)).permute(1, 2, 0).numpy()

        return img, target



