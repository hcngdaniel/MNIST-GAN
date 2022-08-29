#!/usr/bin/env python3
import cv2
import torch
from model import MNISTGAN


model_save_name = 'v3'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
model = MNISTGAN()
model.to(device)
model.load_state_dict(torch.load(f"saves/{model_save_name}.pth", map_location=device))
model.eval()


def show_fake_img():
    noise = torch.randn((1, 100), device=device)
    img = model.generator(noise)[0].cpu().detach().numpy()
    img = (img + 1) / 2
    cv2.imshow("generated", img)
    key = cv2.waitKey(-1)
    if key == 27:
        exit()


while True:
    show_fake_img()
