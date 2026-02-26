import os
from .sd3_vae_encoder import SD3VAEVisionTower
from unipath.model.pixcell_dit import PixCellTransformer2DConfig, PixCellTransformer2D
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def build_gen_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'gen_vision_tower')
    if "sd3" in vision_tower.lower() or "stable-diffusion-3" in vision_tower.lower():
        return SD3VAEVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_dit(vision_tower_cfg, **kwargs):
    if not hasattr(vision_tower_cfg, "hidden_size"):
        if "3B" in vision_tower_cfg.model_name_or_path:
            vision_tower_cfg.hidden_size = 2048
        elif "7B" in vision_tower_cfg.model_name_or_path:
            vision_tower_cfg.hidden_size = 3584

    dit = PixCellTransformer2D(
        PixCellTransformer2DConfig(condition_channels=vision_tower_cfg.hidden_size)
    )

    flow_matching_scheduler_id = getattr(
        vision_tower_cfg,
        "flow_matching_scheduler",
        "Alpha-VLLM/Lumina-Next-SFT-diffusers",
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        flow_matching_scheduler_id,
        subfolder="scheduler",
    )
    return dit, noise_scheduler