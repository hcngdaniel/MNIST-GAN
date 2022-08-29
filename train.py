#!/usr/bin/env python3
import os.path

import yaml
from types import SimpleNamespace
import cv2

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from model import MNISTGAN


# load configs
with open('train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    config = SimpleNamespace(**config)
    config.lr = eval(config.lr)

# load device
device = config.device
if 'cuda' in device and not torch.cuda.is_available():
    device = 'cpu'
    print(f'{device} is not available, switched to cpu')
device = torch.device(device)

# load dataset
dataset = MNIST(
    root='.',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]),
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
)

# load model
gan = MNISTGAN()
gan.to(device)
if os.path.exists(f'saves/{config.save}.pth'):
    gan.load_state_dict(torch.load(f'saves/{config.save}.pth', map_location=device))
generator = gan.generator
discriminator = gan.discriminator


# define loss fn
def generator_loss_fn(fake_out):
    return -torch.mean(fake_out)


def discriminator_loss_fn(real_out, fake_out):
    return -torch.mean(real_out) + torch.mean(fake_out)


# define optimizers
generator_optim = optim.RMSprop(
    generator.parameters(),
    lr=config.lr,
)
discriminator_optim = optim.RMSprop(
    discriminator.parameters(),
    lr=config.lr,
)

# define targets
real_target_conf = torch.ones((config.batch_size, 1), device=device)
fake_target_conf = torch.zeros((config.batch_size, 1), device=device)


# train
def train():
    for batch, (real_samples, labels) in enumerate(dataloader, 1):
        # config input
        real_samples = real_samples.to(device)
        noise = torch.randn((config.batch_size, 100), device=device)
        fake_samples = generator(noise).detach()

        # train discriminator
        # calculate loss
        discriminator_loss = discriminator_loss_fn(
            discriminator(real_samples),
            discriminator(fake_samples)
        )

        # optimize discriminator
        discriminator_optim.zero_grad()
        discriminator_loss.backward()
        discriminator_optim.step()

        # clamp weights
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        # train generator
        if batch % config.n_critic == 0:
            # generate
            generateds = generator(noise)

            # calculate loss
            generator_loss = generator_loss_fn(discriminator(generateds))

            # optimize generator
            generator_optim.zero_grad()
            generator_loss.backward()
            generator_optim.step()

            # show image
            if batch // config.n_critic % 10 == 0:
                img = generateds[0].cpu().detach()
                img = (img + 1) / 2
                img = img.numpy()
                cv2.imshow("generated", img)

            # print loss
            print(f"discriminator loss: {discriminator_loss.cpu().detach().numpy()}    "
                  f"generator loss: {generator_loss.cpu().detach().numpy()}")

        # pause
        cv2.waitKey(1)


# main train loop
for epoch in range(config.start_epoch, config.end_epoch):
    train()
    if epoch % 2 == 0:
        torch.save(gan.state_dict(), f'saves/{config.save}.pth')
        print('saved')
