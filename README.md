# GAN-based-LF-IQA
Light Field (LF) cameras capture spatial and angular information of a scene, generating a high-dimensional data that brings several challenges to compression, transmission, and reconstruction algorithms. One research area that has been attracting a lot of attention is the design of Light Field images quality assessment (LF-IQA) methods. In this paper, we propose a NoReference (NR) LF-IQA method that is based on reference-free distortion maps. With this goal, we first generate a synthetically distorted dataset of 2D images. Then, we compute SSIM distortion maps of these images and use these maps as ground error maps. We train a GAN architecture using these SSIM distortion maps as quality labels. This trained model is used to generate reference-free distortion maps of sub-aperture images of LF contents. Finally, the quality prediction is obtained performing the following steps: 1) perform a non-linear dimensionality reduction with a isometric mapping of the generated distortion
maps to obtain the LFI feature vectors and 2) perform a regression using a Random Forest Regressor (RFR) algorithm to obtain the LF quality estimates. Results show that the proposed method is robust and accurate, outperforming several state-of-the-art LF-IQA methods. 

## Paper: 
Paper can be found [HERE](https://www.frontiersin.org/articles/10.3389/frsip.2022.815058/full). 

## Code:
GAN: [pix-to-pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
Augmentation Library: [Albumentation](https://albumentations.ai/)

## Distorted COCOStuff Dataset:
Coming soon...
