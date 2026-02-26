import warnings

from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from unipath.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    # Import on demand so non-inference backends are not required at import time.
    from unipath.model.language_model.blip3o_qwen_inference import blip3oQwenForInferenceLM

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'


    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model = blip3oQwenForInferenceLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16).to('cuda:0')

    # 确保推理时 gen_vision_tower（SD3 VAE）被加载并迁移到与主模型一致的设备/精度
    try:
        gen_vision_tower = model.get_gen_vision_tower()
        if gen_vision_tower is not None:
            # 延迟加载的 VAE 在首次使用前加载；随后再迁移设备/精度
            gen_vision_tower.load_model()
            target_device = next(model.parameters()).device
            target_dtype = next(model.parameters()).dtype
            gen_vision_tower.to(device=target_device, dtype=target_dtype)
    except Exception as e:
        warnings.warn(f"Failed to place gen_vision_tower on GPU for inference: {e}")

    image_processor = None
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
