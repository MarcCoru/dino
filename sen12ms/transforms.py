import skimage.exposure
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
#import torchvision.transforms as T

bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11", "S2B12"]

def get_rgb(input):
    """
    picks RGB bands from the 12 Sentinel bands
    :param input: 12-band image tensor
    :return: 3-band RGB tensor
    """
    rgb_band_idxs = [bands.index(b) for b in ["S2B4", "S2B3", "S2B2"]] # could be also hardcoded as [3,2,1]
    return input[rgb_band_idxs]

def get_falsecolor(input):
    """
    picks false color bands from the 12 Sentinel bands (for visual interpretation of vegetation)
    :param input: 12-band image tensor
    :return: 3-band NIR-RED-GREEN tensor
    """
    rgb_band_idxs = [bands.index(b) for b in ["S2B8", "S2B4", "S2B3"]]
    return input[rgb_band_idxs]


def equalize_hist(input):
    """a wrapper around skimage.exposure.equalize_hist to perform histogram equalization"""
    return np.float32(skimage.exposure.equalize_hist(input.numpy()))

def get_transform(fold="train", mode="light", cropsize=96):
    """
    :param fold: "train", "val", "test"
    :param mode: none, light, heavy
        none: only base transforms: 3-band RGB calculation and normalization
        light: base transforms + random crop + flip + rotate + randbrightness + randomcotrast
        heavy: light + using RGB + NIR-RED-GREEN bands + channel shuffle + heavier color jitter
    :return: transform function of format: input(3-band), target = transform(input(12-band), target)
    """
    assert fold in ["train", "val", "test"]
    assert mode in ["none", "light", "heavy"]

    basetransforms = A.Compose([
        A.Normalize(mean=(0.4142, 0.4722, 0.6359),
                    std=(0.2344, 0.2008, 0.1544),
                    max_pixel_value=1.),
        A.Resize(224, 224)])

    """Setup albumentation transforms"""
    traintransforms = A.Compose([
        basetransforms,
        #A.RandomCrop(cropsize, cropsize),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.5),
        ToTensorV2()
    ], p=1)

    traintransformsheavy = A.Compose([
        basetransforms,
        #A.RandomResizedCrop(cropsize, cropsize, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4, p=0.5),
        A.ChannelShuffle(p=0.2),
        ToTensorV2()
    ], p=1)

    valtransforms = A.Compose([
        basetransforms,
        #A.RandomCrop(cropsize, cropsize),
        ToTensorV2()
    ], p=1)

    testtransforms = A.Compose([
        basetransforms,
        #A.CenterCrop(cropsize, cropsize),
        ToTensorV2()
    ], p=1)

    nonetransforms = A.Compose([basetransforms,
                                ToTensorV2()], p=1)

    """Construct transforms function"""
    def transform(input, target):
        #if mode == "heavy":
        #    input = np.random.choice([get_rgb, get_falsecolor])(input)
        #elif mode in ["none", "light"]:
        #    input = get_rgb(input)
        #input = equalize_hist(input) # this could be probably done in albumentations as well

        return input, target

    return transform
