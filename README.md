# GAN-based-LF-IQA
In this paper, we propose a NoReference (NR) LF-IQA method that is based on reference-free distortion maps. With this goal, we first generate a synthetically distorted dataset of 2D images. Then, we compute SSIM distortion maps of these images and use these maps as ground error maps. We train a GAN architecture using these SSIM distortion maps as quality labels. This trained model is used to generate reference-free distortion maps of sub-aperture images of LF contents. Finally, the quality prediction is obtained performing the following steps: 1) perform a non-linear dimensionality reduction with a isometric mapping of the generated distortion
maps to obtain the LFI feature vectors and 2) perform a regression using a Random Forest Regressor (RFR) algorithm to obtain the LF quality estimates.

## Paper: 
[Blind visual quality assessment of light field images based on distortion maps](https://www.frontiersin.org/articles/10.3389/frsip.2022.815058/full). 

## Code:
- The code in file `ssim_and_ab_pair.py` first generates augmented images using ALbumentation library, SSIM error maps, and then creates AB pair using error maps and corresponding distorted / augmented images of [CocoStuff](https://github.com/nightrome/cocostuff). 
- GAN: [pix-to-pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Augmentation Library: [Albumentation](https://albumentations.ai/)

## Distorted COCOStuff Dataset:
Coming soon...

## Cite this article:
```
@ARTICLE {lfiqaGANfront2022,
    author  = "Sana Alamgeer, and Mylène C.Q. Farias",
    title   = "Blind visual quality assessment of light field images based on distortion maps",
    journal = "Frontiers Signal Processing",
    year    = "2022",
    month   = "August"
}
```
