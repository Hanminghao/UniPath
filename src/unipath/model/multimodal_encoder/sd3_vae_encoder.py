import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from typing import Optional, Union, Dict
from functools import partial, reduce
from PIL import Image
import numpy as np
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)


class SD3VAEImageProcessor:
    def __init__(self, 
                 size=(384, 384), 
                 crop_size: Dict[str, int] = None, 
                 resample=PILImageResampling.BICUBIC, 
                 rescale_factor=1 / 255.0,
                 data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": size[0], "width": size[1]}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors="pt"):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        try:
            transforms = [
                convert_to_rgb,
                to_numpy_array,
                partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
                partial(self._rescale_and_normalize, scale=self.rescale_factor, data_format=self.data_format),
                partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
            ]

            images = reduce(lambda x, f: [*map(f, x)], transforms, images)
            data = {"pixel_values": images}
        except ValueError as e:
            try:
                transforms = [
                    convert_to_rgb,
                    to_numpy_array,
                    partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
                    partial(self._rescale_and_normalize, scale=self.rescale_factor, data_format=self.data_format),
                    partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
                ]
                images = reduce(lambda x, f: [*map(f, x)], transforms, images)
                processed_images = [np.repeat(img, repeats=3, axis=0) for img in images]
                data = {"pixel_values": processed_images}
            except ValueError as e:
                print(f"SD3VAE 灰度图像处理失败: {e}")
                raise

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _rescale_and_normalize(self, image, scale, data_format):
        if image.max() > 1.0:
            image = image * scale
        image = image * 2.0 - 1.0
        return image

    @property
    def crop_size_dict(self):
        return self.crop_size

    @property
    def size_dict(self):
        return {"height": self.size[0], "width": self.size[1]}


class SD3VAEVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        
        self.model_name = "stabilityai/stable-diffusion-3.5-large"
        self.subfolder = "vae"
        
        image_size = getattr(vision_tower_cfg, 'image_size', 384)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_processor = SD3VAEImageProcessor(
            size=image_size,
            crop_size={"height": image_size[0], "width": image_size[1]}
        )
        
        print(f"SD3VAE configured for image size: {image_size}")

        if not delay_load:
            print(f"Loading SD3 VAE: {self.model_name}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = True

    def load_model(self, device_map=None, torch_dtype=None):
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.model_name))
            return

        print(f"Loading AutoencoderKL from {self.model_name}, subfolder: {self.subfolder}")
        
        load_kwargs = {
            "subfolder": self.subfolder,
        }
        
        if device_map is not None:
            load_kwargs["device_map"] = device_map
            
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
            
        self.vision_tower = AutoencoderKL.from_pretrained(
            self.model_name, 
            **load_kwargs
        )
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        
        if isinstance(images, (list, tuple)) and len(images) > 0:
            if isinstance(images[0], Image.Image):
                processed = self.image_processor.preprocess(images, return_tensors="pt")
                images = torch.tensor(processed["pixel_values"])
            elif isinstance(images[0], torch.Tensor):
                images = torch.stack(images, dim=0)
        elif isinstance(images, Image.Image):
            processed = self.image_processor.preprocess([images], return_tensors="pt")
            images = torch.tensor(processed["pixel_values"])
        elif isinstance(images, torch.Tensor):
            if images.max() > 1.0:
                images = images / 255.0
                images = images * 2.0 - 1.0
        
        if hasattr(self.vision_tower, 'device'):
            images = images.to(device=self.vision_tower.device)
        if hasattr(self.vision_tower, 'dtype'):
            images = images.to(dtype=self.vision_tower.dtype)
            
        with torch.no_grad():
            latent_dist = self.vision_tower.encode(images)
            latents = latent_dist.latent_dist.sample()
            
            latents = latents * self.vision_tower.config.scaling_factor
            
        return latents

    def decode(self, latents, normalize=True, return_tensor=False):
        if not self.is_loaded:
            self.load_model()
        
        if hasattr(self.vision_tower, 'device'):
            latents = latents.to(device=self.vision_tower.device)
        
        if hasattr(self.vision_tower, 'dtype'):
            latents = latents.to(dtype=self.vision_tower.dtype)
        
        with torch.no_grad():
            latents = latents / self.vision_tower.config.scaling_factor
            
            if hasattr(self.vision_tower.config, 'shift_factor') and self.vision_tower.config.shift_factor is not None:
                latents = latents - self.vision_tower.config.shift_factor
            
            samples = self.vision_tower.decode(latents).sample
        
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        
        if return_tensor:
            return samples
        
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        
        from diffusers.utils import numpy_to_pil
        images = numpy_to_pil(samples)
        
        return images

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self.is_loaded and hasattr(self.vision_tower, 'dtype'):
            return self.vision_tower.dtype
        elif self.is_loaded:
            try:
                first_param = next(self.vision_tower.parameters())
                return first_param.dtype
            except (StopIteration, AttributeError):
                pass
        return torch.float32

    @property
    def device(self):
        if hasattr(self.vision_tower, 'device'):
            return self.vision_tower.device
        return torch.device('cpu')

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            class DummyConfig:
                def __init__(self):
                    self.hidden_size = 16
                    self.scaling_factor = 1.5305
            return DummyConfig()

    @property
    def hidden_size(self):
        return self.config.latent_channels if hasattr(self.config, 'latent_channels') else 16

    def to(self, *args, **kwargs):
        """Override to method to handle dtype and device changes"""
        result = super().to(*args, **kwargs)
        
        if self.is_loaded and hasattr(self, 'vision_tower'):
            self.vision_tower = self.vision_tower.to(*args, **kwargs)
            
        return result
        
    def train(self, mode=True):
        if self.is_loaded:
            self.vision_tower.eval()
        return super().train(False)
