# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:41:17 2022

@author: sanaalamgeer
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from os import listdir
import os
import albumentations as A
import sys
import numpy
from scipy.signal import fftconvolve
from scipy import ndimage
import gauss
import albumentations.augmentations.geometric.transforms as B
#%%
def read_image(url):
	image = imageio.imread(url)
	if image.ndim < 3:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	return image

def augment_and_return(aug, image):
	aimg = aug(image=image)['image']
	return aimg

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """	    
    if img1.ndim >= 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim >= 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
		
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 5 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = fftconvolve(img1, window, mode='valid')
    mu2 = fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
        ssim_map = cv2.normalize(ssim_map, None, 0, 250, cv2.NORM_MINMAX)
        ssim_map = np.uint8(ssim_map)
	    
        ssim_map = cv2.cvtColor(ssim_map,cv2.COLOR_GRAY2RGB)
        ssim_map = cv2.resize(ssim_map, (img1.shape[1], img1.shape[0]))		
        return ssim_map
	
def save_aug_and_ssim(img1, img2):
	im_AB = numpy.concatenate([img1, img2], 1)
	im_AB = cv2.resize(im_AB, (512, 256))	
	return im_AB

def save_img_AB(img, imgName, destination):
	output = destination + imgName + '.jpg'	
	imageio.imwrite(output, img)
#%%
inputPath = '/mnt/nas/sanaalamgeer/Projects/5/dataset/cocostuff/'
outputPath = '/mnt/nas/sanaalamgeer/Projects/5/dataset/train/'

input_images = [f for f in listdir(inputPath)] 
input_images.sort()
#%%
#calling function
#aug = A.HorizontalFlip(p=1) #0.5 - smaller the value, minor the change is.
#'''
listagF = {"HorizontalFlip": "A.HorizontalFlip(p=1)",
		   "VerticalFlip": "A.VerticalFlip(p=1)",
		   "GaussNoise": "A.GaussNoise(p=1)",
		   "Perspective": "A.Perspective(p=1)",
		   "PiecewiseAffine": "B.PiecewiseAffine(p=1)",
		   "Sharpen": "A.Sharpen(p=1)",
		   "Superpixels": "A.Superpixels(p=1)",
		   "Emboss": "A.Emboss(p=1)",
		   "CLAHE": "A.CLAHE(p=1)",
		   "Rotate": "A.Rotate(p=1)",
		   "ShiftScaleRotate": "A.ShiftScaleRotate(p=1)",
		   "Blur": "A.Blur(p=1)",
		   "OpticalDistortion": "A.OpticalDistortion(p=1)",
		   "GridDistortion": "A.GridDistortion(p=1)",
		   "HueSaturationValue": "A.HueSaturationValue(p=1)",
		   "MotionBlur": "A.MotionBlur(p=1)",
		   "MedianBlur": "A.MedianBlur(p=1)",
		   "RandomFog": "A.RandomFog(p=1)",
		   "RandomBrightnessContrast": "A.RandomBrightnessContrast(p=1)",
		   "RandomGamma": "A.RandomGamma(p=1)",
		   "InvertImg": "A.InvertImg(p=1)",
		   "RandomSunFlare": "A.RandomSunFlare(p=1)",
		   "RandomGridShuffle": "A.RandomGridShuffle(p=1)",
		   "RandomToneCurve": "A.RandomToneCurve(p=1)",
		   "RandomRain": "A.RandomRain(p=1)",
		   "ImageCompression": "A.ImageCompression(p=1)",
		   "FancyPCA": "A.FancyPCA(p=1)",
		   "ElasticTransform": "B.ElasticTransform(p=1)",
		   "Equalize": "A.Equalize(p=1)"}
#'''
#listagF = {"ToTensor": "A.ToTensor(p=1)"}
keys = list(listagF.keys())
values = list(listagF.values())

#%%
count = 0
for k in range(len(keys))[:]:
	print(f'{keys[k]} - {k}/{len(keys)}')
	
	#converting string to a function call
	aug = eval(values[k])
	
	#saving key to path to output folder
	folder = keys[k] 
	
	for i in input_images[:]:
		#print(i)
		path2img = inputPath + i
		image = read_image(path2img)
		
		#print("-------{} - Augmenting---------".format(la))
		aimg = augment_and_return(aug, image)
		if aimg.ndim < 3:
			aimg = cv2.cvtColor(aimg, cv2.COLOR_GRAY2RGB)
		aimg = cv2.resize(aimg, (image.shape[1], image.shape[0]))
				
		ssim_map = ssim(image, aimg)
		
		img_AB = save_aug_and_ssim(aimg, ssim_map)
		
		save_img_AB(img_AB, str(count), outputPath)
		
		count += 1
		
#####################END################################
