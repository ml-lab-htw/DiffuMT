from PIL import Image
import torch

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

class ScaleSeg(object):
        """Map a segmentation mask to the seg-channel scale the UNet was TRAINED on.

        The model was trained on {0,1} label PNGs run through ToTensor (which
        divides by 255), so its conditioning channel is {0, 1/255 ~= 0.0039}.
        Feeding {0, 1.0} instead is a 255x-too-strong signal that pushes the
        model out of distribution (dark, grainy, thick-filament samples).
        This transform reproduces the training scale regardless of whether the
        stored mask is {0,1} or {0,255}: binarise, then divide by 255 -> {0,1/255}.
        Ground truth: SynthMT-Studio/app.py `_prep_mask` = (arr>0).float()/255.
        (This project uses a single binary foreground class.)
        """
        def __call__(self, tensor):
            return (tensor > 0).float() / 255.0

        def __repr__(self):
            return self.__class__.__name__ + '()'


class AddGaussianNoise(object):
        """
        Adds Gaussian noise to a tensor.
        Used to simulate 'dirty' microscopy backgrounds during training (Run B).
        """
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean
            
        def __call__(self, tensor):
            # Add noise: x = x + N(mean, std)
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)