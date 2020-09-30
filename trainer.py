
import shutil
import copy
import os
import time
import datetime
import json
import yaml
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms

from skimage.io import imsave

from utils import ImageDataset, ImagePool, update_average, compute_grad2
from model import Generator, Discriminator
import losses

class Trainer():
    def __init__(self, params):
        self.params = params
        self.dataset_name = params["dataset_name"]
        self.data_dir = "./data"

        self.max_epoch = params["max_epoch"]
        self.interval = params["interval"]
        self.batch_size = params["batch_size"]
        self.image_size = params["image_size"]
        self.latent_dim = params["latent_dim"]

        self.learning_rate = params["learning_rate"]
        self.use_cr = params["use_cr"]
        self.start_cr = params["start_cr"]
        self.use_ema = params["use_ema"]
        self.ema_beta = 0.999
        self.use_zero_centered_grad = True
        self.unet_dis = params["unet_dis"]
        
        self.num_tiles = params["num_tiles"]

        dt_now = datetime.now()
        dt_seq = dt_now.strftime("%y%m%d_%H%M")
        self.result_dir = os.path.join("./result", f"{dt_seq}_{self.dataset_name}")
        self.weight_dir = os.path.join(self.result_dir, "weights")
        self.sample_dir = os.path.join(self.result_dir, "sample")
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        pool_size = self.batch_size * 10
        self.image_pool = ImagePool(pool_size)

        with open(os.path.join(self.result_dir, "params.yml"), mode="w") as f:
            yaml.dump(params, f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self, resume_from=None):
        print("Construct generator and discriminator")
        generator = Generator(self.params)
        discriminator = Discriminator(self.params)
        print(generator)
        print(discriminator)
        generator.to(self.device)
        discriminator.to(self.device)
        if self.use_ema:
            generator_ema = copy.deepcopy(generator)
            update_average(generator_ema, generator, 0)
            print("Use Exponential Moving Average")

        if self.use_cr:
            print("Use Consistency-Regularization")
        else:
            print("Not use Consistency-Regularization")

        print("Construct dataloader")
        train_dataset = ImageDataset(self.params)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=self.batch_size,
                                      num_workers=4,
                                      shuffle=True,
                                      drop_last=True)

        # construct optimizers
        optimizer_G = Adam(generator.parameters(),
                           lr=self.learning_rate,
                           betas=(0, 0.999))
        optimizer_D = Adam(discriminator.parameters(),
                            lr=self.learning_rate*4, 
                            betas=(0, 0.999))

        criterion = losses.HingeLoss(self.batch_size, self.device)

        start_epoch = 1
        fixed_z = None

        if resume_from:
            file_list = os.listdir(f"{resume_from}/weights")
            epoch_list = list(set([int(file.split("_")[0]) for file in file_list]))
            latest_epoch = max(epoch_list)
            start_epoch = latest_epoch + 1
            self.max_epoch += latest_epoch
            generator.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_generator.pth"))
            discriminator.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_discriminator.pth"))
            optimizer_G.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_optG.pth"))
            optimizer_D.load_state_dict(torch.load(f"{resume_from}/weights/{latest_epoch}_optD.pth"))
            fixed_z = torch.load(f"{resume_from}/fixed_z.pth")
            shutil.rmtree(self.result_dir)
            self.result_dir = resume_from
            self.weight_dir = os.path.join(self.result_dir, "weights")
            self.sample_dir = os.path.join(self.result_dir, "sample")
            print(f"Resume training from {resume_from} at epoch {start_epoch}")
        
        if fixed_z is None:
            fixed_z = truncated_randn(self.num_tiles**2, self.latent_dim).to(self.device)
            torch.save(fixed_z, f"{self.result_dir}/fixed_z.pth")

        for epoch in range(start_epoch, self.max_epoch + 1):
            print(f"epoch {epoch} start")
            mix_prob = 0.5 * min((epoch - 1), self.start_cr) / self.start_cr
            for i, batch in enumerate(train_dataloader):
                if self.use_cr:
                    real_images, transformed_real_images = batch
                    real_images = real_images.to(self.device)
                    transformed_real_images = transformed_real_images.to(self.device)
                else:
                    real_images = batch
                    real_images = real_images.to(self.device)
                real_images.requires_grad_()

                z = torch.randn(real_images.shape[0], self.latent_dim).to(self.device)
                fake_images = generator(z).detach()
                if not self.unet_dis:
                    out_real = discriminator(real_images)
                    out_fake = discriminator(fake_images)
                    loss_d_real = criterion.dis_real_loss(out_real)
                    loss_d_fake = criterion.dis_fake_loss(out_fake)
                    loss_d = loss_d_real + loss_d_fake
                    if self.use_cr:
                        out_t_real = discriminator(transformed_real_images)
                        loss_d_cr = torch.mean(torch.sqrt((out_real - out_t_real)**2 + 1e-8))
                        loss_d += 0.01 * loss_d_cr
                else:
                    out_enc_real, out_dec_real = discriminator(real_images)
                    out_enc_fake, out_dec_fake = discriminator(fake_images)
                    loss_d = criterion.dis_real_loss(out_enc_real) + \
                                criterion.dis_real_loss(out_dec_real) + \
                                criterion.dis_fake_loss(out_enc_fake) + \
                                criterion.dis_fake_loss(out_dec_fake)

                    if np.random.rand() < mix_prob:
                        mask_tensor = generate_mask(self.batch_size, self.image_size).to(self.device)
                        cutmix_inputs = cutmix(real_images, fake_images, mask_tensor)
                        out_enc_cutmix, out_dec_cutmix = discriminator(cutmix_inputs)
                        loss_enc = criterion.dis_fake_loss(out_enc_cutmix)
                        loss_dec = criterion.dis_cutmix_loss(out_dec_cutmix, mask_tensor)
                        loss_d += loss_enc + loss_dec                       
                        
                        cutmix_outputs = cutmix(out_dec_real, out_dec_fake, mask_tensor)

                        _, out_cutmix_inputs = discriminator(cutmix_inputs)
                        loss_d_cr = torch.mean(torch.sqrt((out_cutmix_inputs - cutmix_outputs) ** 2 + 1e-12))
                        loss_d += loss_d_cr

                    out_real = out_enc_real
                
                if self.use_zero_centered_grad:
                    loss_d += 0.1 * compute_grad2(out_real, real_images).mean()

                optimizer_D.zero_grad()
                loss_d.backward()
                optimizer_D.step()
                
                z = torch.randn(real_images.shape[0], self.latent_dim).to(self.device)
                fake_images = generator(z)
                if not self.unet_dis:
                    loss_g = criterion.gen_loss(discriminator(fake_images))
                else:
                    out_enc, out_dec = discriminator(fake_images)
                    loss_g_enc = criterion.gen_loss(out_enc)
                    loss_g_dec = criterion.gen_loss(out_dec)
                    loss_g = loss_g_enc + loss_g_dec

                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()

                if self.use_ema:
                    update_average(generator_ema, generator, self.ema_beta)

                print(f"epoch {epoch} {i}/{len(train_dataloader)}, loss_D: {loss_d.item():.4f}, loss_g: {loss_g.item():.4f}")

            if epoch % self.interval == 0:
                torch.save(generator.state_dict(), f"{self.weight_dir}/{epoch}_generator.pth")
                torch.save(discriminator.state_dict(), f"{self.weight_dir}/{epoch}_discriminator.pth")
                torch.save(optimizer_G.state_dict(), f"{self.weight_dir}/{epoch}_optG.pth")
                torch.save(optimizer_D.state_dict(), f"{self.weight_dir}/{epoch}_optD.pth")
            if self.use_ema:
                torch.save(generator_ema.state_dict(), f"{self.weight_dir}/{epoch}_generator_ema.pth")
        
                filename = f"{epoch}_gen_ema.png"
                generator_ema.eval()
                with torch.no_grad():
                    sample_images = generator_ema(fixed_z)
                    save_samples(sample_images, self.sample_dir, filename, self.image_size, self.num_tiles)
                generator_ema.train()

            generator.eval()
            filename = f"{epoch}_gen.png"
            with torch.no_grad():
                sample_images = generator(fixed_z)
                save_samples(sample_images, self.sample_dir, filename, self.image_size, self.num_tiles)
            generator.train()


def cutmix(images_1, images_2, mask):
    cutmix_image = (images_1 + 1.) * mask + (images_2 + 1.) * (1. - mask) - 1
    return cutmix_image


def generate_mask(batch_size, image_size):
    result_tensor = torch.zeros((batch_size, 1, image_size, image_size))
    for i in range(batch_size):
        W = H = image_size
        cut_rat = np.sqrt(1. - np.random.beta(2, 2))
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        result_tensor[i, :, bby1:bby2, bbx1:bbx2] = 1

    return result_tensor


def save_samples(image_tensor, sample_dir, filename, image_size, num_tiles):
    img = image_tensor.to("cpu").detach()
    img = img.numpy().transpose(0, 2, 3, 1)
    result = np.zeros((image_size*num_tiles, image_size*num_tiles, 3))
    img = (img + 1) * 127.5
    
    for i in range(num_tiles):
        for j in range(num_tiles):
            result[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, :] = img[num_tiles*i+j, :, :, :]
    result = result.astype(np.uint8)
    imsave(f"{sample_dir}/{filename}", result)


def truncated_randn(bs, latent_dim, th=0.8):
    while True:
        temp_z = np.random.randn(bs*latent_dim*10)
        z = temp_z[np.abs(temp_z) < th]
        if z.size >= bs * latent_dim:
            z = z[:bs*latent_dim]
            z = z.reshape(bs, latent_dim)
            return torch.FloatTensor(z)
