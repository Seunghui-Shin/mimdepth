# Copyright © NavInfo Europe 2023.
# Adapted from SimMIM and BEIT.

import numpy as np
import cv2
import copy
import math
import random
import torch

import os
from PIL import Image
from IPython.display import Image as Img
from IPython.display import display

import matplotlib.pyplot as plt

from imagecorruptions import corrupt, get_corruption_names

import torch
#from transformers import DPTForDepthEstimation, DPTImageProcessor
import requests
import pickle
#model_name = 'Intel/dpt-hybrid-midas'
#model = DPTForDepthEstimation.from_pretrained(model_name, output_attentions=True).to('cuda')
#feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
image_size = (640,192)
filter_size = 10

class F_MaskGenerator:
    def __init__(self, img, img_shape=(192, 640), mask_patch_size=32, model_patch_size=4, mask_ratio=0.25,
                 mask_strategy='random', temperature=10000, frequency = 'high', idx = 0):
        self.img = img
        self.img_shape = img_shape
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.temperature = temperature
        self.frequency = frequency
        self.idx = int(idx)

        assert self.img_shape[0] % self.mask_patch_size == 0
        assert self.img_shape[1] % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.token_shape = np.zeros(len(self.img_shape), dtype=int)
        self.token_shape[0] = self.img_shape[0] // self.mask_patch_size
        self.token_shape[1] = self.img_shape[1] // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.token_shape[0] * self.token_shape[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
        # High-pass filter 생성
        center = self.mask_patch_size // 2
        self.high_pass_filter = np.zeros((self.mask_patch_size, self.mask_patch_size))
        for i in range(self.mask_patch_size):
            for j in range(self.mask_patch_size):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance > filter_size:  # 원의 반지름을 조절하여 고주파 정보를 제어할 수 있습니다.
                    self.high_pass_filter[i, j] = 1

    def random_masking(self):
        mask = np.zeros(shape=self.token_count, dtype=int)
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask[mask_idx] = 1

        mask = mask.reshape((self.token_shape[0], self.token_shape[1]))
        return mask

    def blockwise_masking(self, min_num_mask_patches=16,
                          min_blockwise_aspect=0.3):
        mask = np.zeros(shape=self.token_count, dtype=int)
        mask = mask.reshape((self.token_shape[0], self.token_shape[1]))
        num_tokens_masked = 0
        NUM_TRIES = 10
        max_blockwise_aspect = 1 / min_blockwise_aspect
        log_aspect_ratio = (math.log(min_blockwise_aspect), math.log(max_blockwise_aspect))
        while num_tokens_masked < self.mask_count:
            max_mask_patches = self.mask_count - num_tokens_masked

            delta = 0
            for attempt in range(NUM_TRIES):
                target_area = random.uniform(min_num_mask_patches, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if h < self.token_shape[0] and w < self.token_shape[1]:
                    top = random.randint(0, self.token_shape[0] - h)
                    left = random.randint(0, self.token_shape[1] - w)

                    num_masked = mask[top: top + h, left: left + w].sum()
                    # Overlap
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1

                    if delta > 0:
                        break

            if delta == 0:
                break
            else:
                num_tokens_masked += delta

        return mask

    def __call__(self):
        if self.mask_strategy == 'random':
            mask = self.random_masking()
        elif self.mask_strategy == 'blockwise':
            mask = self.blockwise_masking()
        elif self.mask_strategy == 'f_random':
            mask = self.f_masking(self.img, self.mask_patch_size, self.temperature, self.frequency)
        elif self.mask_strategy == 'f_block':
            mask = self.f_blockwise_masking(self.img, self.mask_patch_size, self.temperature, self.frequency)
        elif self.mask_strategy == 'attention':
            mask = self.att_masking(self.img, self.mask_patch_size, self.frequency)
        else:
            raise NotImplementedError
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask
    
    def f_masking(self, img, mask_patch_size, temperature = 10000, frequency='low'):
        # 이미지 크기 및 패치 크기 가져오기
        height, width, channels = img.shape
        h_patches = height // mask_patch_size
        w_patches = width // mask_patch_size

        patch_magnitudes = []

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for y in range(0, height, mask_patch_size):
            for x in range(0, width, mask_patch_size):
                patch = gray_image[y:y+mask_patch_size, x:x+mask_patch_size]
                patch_fft = torch.fft.fft2(torch.from_numpy(patch))
                patch_fft = torch.fft.fftshift(patch_fft)
                magnitude = torch.log1p(torch.abs(patch_fft))
                if frequency == 'low':
                    patch_magnitudes.append(-magnitude)
                else:
                    patch_magnitudes.append(magnitude)

        magnitudes = [torch.sum(magnitude* self.high_pass_filter) for magnitude in patch_magnitudes]
        softmax_values = softmax(magnitudes, temperature)

        mask_idx = np.random.choice(self.token_count, self.mask_count, replace=False, p=softmax_values)

        mask = np.zeros(shape=self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.token_shape[0], self.token_shape[1]))
        return mask
    
    def f_blockwise_masking(self, img, mask_patch_size, temperature = 10000, frequency='low', min_num_mask_patches=16,
                          min_blockwise_aspect=0.3):
        # 이미지 크기 및 패치 크기 가져오기
        height, width, channels = img.shape
        h_patches = height // mask_patch_size
        w_patches = width // mask_patch_size

        patch_magnitudes = []

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for y in range(0, height, mask_patch_size):
            for x in range(0, width, mask_patch_size):
                patch = gray_image[y:y+mask_patch_size, x:x+mask_patch_size]
                patch_fft = torch.fft.fft2(torch.from_numpy(patch))
                patch_fft = torch.fft.fftshift(patch_fft)
                magnitude = torch.log1p(torch.abs(patch_fft))
                if frequency == 'low':
                    patch_magnitudes.append(-magnitude)
                else:
                    patch_magnitudes.append(magnitude)

        magnitudes = [torch.sum(magnitude* self.high_pass_filter) for magnitude in patch_magnitudes]
        softmax_values = softmax(magnitudes, temperature)
        # softmax_vis = softmax_values.reshape((self.token_shape[0], self.token_shape[1]))
        # softmax_vis = cv2.resize(softmax_vis, dsize=image_size, interpolation = cv2.INTER_NEAREST)
        # heatmap = (softmax_vis / softmax_vis.max()).astype('float32')
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('result', image*heatmap/255)
        # #cv2.imshow('soft', cv2.cvtColor(softmax_vis/softmax_vis.max(), cv2.COLOR_GRAY2BGR)*0.3+image*0.7)
        # cv2.waitKey()

        mask = np.zeros(shape=self.token_count, dtype=int)
        mask = mask.reshape((self.token_shape[0], self.token_shape[1]))
        num_tokens_masked = 0
        NUM_TRIES = 10
        max_blockwise_aspect = 1 / min_blockwise_aspect
        log_aspect_ratio = (math.log(min_blockwise_aspect), math.log(max_blockwise_aspect))
        while num_tokens_masked < self.mask_count:
            max_mask_patches = self.mask_count - num_tokens_masked

            delta = 0
            for attempt in range(NUM_TRIES):
                target_area = random.uniform(min_num_mask_patches, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if h < self.token_shape[0] and w < self.token_shape[1]:
                    mask_idx = np.random.choice(self.token_count, 1, p=softmax_values)[0]
                    top = mask_idx // self.token_shape[1] - h // 2 # random.randint(0, self.token_shape[0] - h)
                    left = mask_idx % self.token_shape[1] - w // 2 # random.randint(0, self.token_shape[1] - w)
                    num_masked = mask[max(0, top): min(top + h, self.token_shape[0]), max(0, left): min(left + w, self.token_shape[1])].sum()
                    # Overlap
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(max(0, top), min(top + h, self.token_shape[0])):
                            for j in range(max(0, left), min(left + w, self.token_shape[1])):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1

                    if delta > 0:
                        break

            if delta == 0:
                break
            else:
                num_tokens_masked += delta

        return mask

    def att_masking(self, img, mask_patch_size, att='low'):
        # 이미지 크기 및 패치 크기 가져오기
        height, width, channels = img.shape
        h_patches = height // mask_patch_size
        w_patches = width // mask_patch_size

        with open('/data/sundong/repos/MIMDepth/data/kitti_data.pkl', 'rb') as f:
            kitti_data = pickle.load(f)
        #print(self.idx)
        patch_magnitudes = kitti_data[self.idx]

        if att == 'low':
            mask_idx = np.argsort(patch_magnitudes)[:self.mask_count]
        else:
            mask_idx = np.argsort(patch_magnitudes)[::-1][:self.mask_count]
        #mask_idx = np.random.choice(self.token_count, self.mask_count, replace=False, p=prob)
        mask = np.zeros(shape=self.token_count, dtype=int)
        mask[mask_idx] = 1
        if att == 'hint':
            mask = show_hints(mask, mask, 0.1)
        mask = mask.reshape((self.token_shape[0], self.token_shape[1]))
        return mask

def show_hints(top_masks, masks, show_ratio):

    n_tokens = masks.shape[0]
#    print(n_tokens)
    reveal_tokens = int(show_ratio*n_tokens)

    selected_high = torch.multinomial(torch.tensor(top_masks, dtype=torch.float), reveal_tokens)
    top_masks[selected_high] = 0
    top_masks.reshape((12,40))
    return top_masks

class F_MIMTransform:
    def __init__(self, img, img_size, mask_patch_size, mask_ratio, mask_strategy='random', temperature=10000, frequency='low', idx=0):
        model_patch_size = 16
        #if isinstance(img, Image.Image):
        #    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(np.transpose(img.numpy(), (1,2,0)), cv2.COLOR_RGB2BGR)
        #print(img.shape)
        self.mask_generator = F_MaskGenerator(
            img = img,
            img_shape=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            temperature = temperature,
            frequency = frequency,
            idx = idx
        )

    def __call__(self):
        mask = self.mask_generator()
        return mask

def softmax(x, t):
    x = np.array(x) / t
    exp_x = np.exp(x-np.max(x)) 
    return exp_x / np.sum(exp_x)

def generate_gif(images, gif_name):    
    im = images[0]
    im.save(gif_name, save_all=True, append_images=images[1:],loop=0xff, duration=200)
    # loop 반복 횟수
    # duration 프레임 전환 속도 (500 = 0.5초)
    return

# 이미지를 불러오고 처리하는 함수
def process_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# 모델을 불러오고 어텐션 맵 추출하는 함수
def extract_attention_map(image):
    # 이미지 처리
    inputs = process_image(image).to('cuda')
    # 어텐션 맵 추출
    outputs = model(**inputs)
    attentions = outputs.attentions
    attention_map = torch.mean(attentions[0], dim=1).squeeze(0)

    mask = attention_map[0, 1:] # [196,196]
    width = int(mask.size(-1)**0.5)
    attention_map = mask.reshape(width, width)
    attention_map = cv2.resize(attention_map.cpu().detach().numpy(), image_size)

    return attention_map / attention_map.max()
