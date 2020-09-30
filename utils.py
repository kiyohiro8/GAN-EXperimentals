
import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, Resize, Normalize, RandomCrop,RandomRotation, ColorJitter, ToPILImage
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, params):
        super(ImageDataset, self).__init__()
        data_dir = params["data_dir"]
        dataset_name = params["dataset_name"]
        image_size = params["image_size"]
        self.on_memory = params["on_memory"]

        self.use_cr = params["use_cr"]
        if self.use_cr:
            self.transform_for_cr = Compose([ToPILImage(),
                                             RandomCrop(int(image_size*0.8)),
                                             RandomHorizontalFlip(p=0.5),
                                             ToTensor()])
    
        self.image_files = sorted(glob(os.path.join(data_dir, dataset_name) + '/*.*'))
        self._size_check(image_size)
        random.seed(42)
        random.shuffle(self.image_files)
        self._size_check(image_size=image_size)
        if self.on_memory:
            self.images = [Image.open(f) for f in self.image_files]

        self.transform = Compose([ColorJitter(brightness=0.2, contrast=0.2),
                            RandomRotation(10),
                            Resize(int(image_size*1.1)),
                            RandomCrop(image_size),
                            RandomHorizontalFlip(p=0.5),
                            ToTensor()])

    def __getitem__(self, index):
        if not self.on_memory:
            image = Image.open(self.image_files[index])
        else:
            image = self.images[index]
        if image.mode == "L":
            image = image.convert("RGB")
        if self.use_cr:
            image = self.transform(image)
            t_image = self.transform_for_cr(image)
            image = (image - 0.5) * 2
            t_image = (t_image - 0.5) * 2
            return image, t_image
        else:
            image = self.transform(image)
            image = (image - 0.5) * 2

            return image

    def __len__(self):
        return len(self.image_files)

    def _size_check(self, image_size):
        new_list = []
        for f in self.image_files:
            image = Image.open(f)
            if min(image.width, image.height) >= image_size:
                new_list.append(f)
        self.image_files = new_list


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def push_and_pop(self, image):
        if self.pool_size == 0:
            self.images.append(image)
            return image
        return_images = []
        for i in range(image.size(0)):
            img = torch.unsqueeze(image[i, :, :, :], 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(img)
                return_images.append(img)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images.pop(random_id)
                    self.images.append(img)
                    return_images.append(tmp)
                else:
                    return_images.append(img)
        return_images = torch.cat(return_images, dim=0)
        return return_images
    
    def __len__(self):
        return len(self.images)


def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg