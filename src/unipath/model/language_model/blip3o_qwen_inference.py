from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateOutput
from unipath.model.blip3o_arch import blip3oMetaModel, blip3oMetaForCausalLM
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration
from unipath.constants import UND_IMAGE_TOKEN_IDX


from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class blip3oQwenConfig(Qwen2_5_VLConfig):
    model_type = "blip3o_qwen_inference"

class blip3oQwenModel(blip3oMetaModel, Qwen2_5_VLModel):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(blip3oQwenModel, self).__init__(config)
        from unipath.model.multimodal_projector.builder import build_down_projector
        if not hasattr(self, 'down_projector') or self.down_projector is None:
            self.down_projector = build_down_projector(config)


class blip3oQwenForInferenceLM(Qwen2_5_VLForConditionalGeneration, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "blip3o_qwen_inference"

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                img_indicator,
                _
            ) = self.prepare_inputs_labels_for_understanding(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        user_text: List[str],
        tokenizer: AutoTokenizer,
        retriever: Optional[Any] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_var: Optional[float] = None,
    ):  
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "Alpha-VLLM/Lumina-Next-SFT-diffusers",
            subfolder="scheduler",
        )

        N_QUERY = self.get_n_query()            
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.get_model().device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)  # B x N
        input_ids = torch.cat([input_ids, torch.tensor([[151665]]).to(device)], dim=1)


        text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)


        if pixel_values is not None:
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            pixel_values = pixel_values.type(self.visual.dtype)
            und_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device)[:und_image_idx.sum(), :]


        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1)

        outputs = self.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1][:,-N_QUERY:,:]
        img_hidden_states = hidden_states
        img_hidden_states = self.get_model().down_projector(img_hidden_states)

        user_text_inputs = tokenizer(
            user_text, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        user_text_input_ids = user_text_inputs.input_ids.to(device)
        user_text_attention_mask = user_text_inputs.attention_mask.to(device)

        user_text_embeds = self.get_model().embed_tokens(user_text_input_ids)

        bsz = img_hidden_states.shape[0]
        # 追加 prototype 条件（可选）
        if retriever is not None:
            with torch.no_grad():
                proto_features, proto_mask = retriever.retrieve(
                    prompts=[ut.strip() for ut in user_text],
                    top_k=4,
                    max_samples=getattr(self.get_model().config, 'max_prototype_samples', 16)
                )
                if hasattr(self.get_model(), 'prototype_projector'):
                    proto_features = self.get_model().prototype_projector(proto_features.to(dtype=self.get_model().dtype, device=device))
                # proto_mask: 0/1 -> bool 以复用现有 mask 拼接逻辑
                proto_mask_bool = proto_mask.to(dtype=user_text_attention_mask.dtype, device=user_text_attention_mask.device)

            encoder_hidden_states = torch.cat([user_text_embeds, img_hidden_states, proto_features], dim=1)
            img_mask = torch.ones(bsz, img_hidden_states.shape[1], device=user_text_attention_mask.device, dtype=user_text_attention_mask.dtype)
            encoder_mask = torch.cat([user_text_attention_mask, img_mask, proto_mask_bool], dim=1)
        else:
            encoder_hidden_states = torch.cat([user_text_embeds, img_hidden_states], dim=1)  # [B, T_text+T_img, H]
            img_mask = torch.ones(bsz, img_hidden_states.shape[1], device=user_text_attention_mask.device, dtype=user_text_attention_mask.dtype)
            encoder_mask = torch.cat([user_text_attention_mask, img_mask], dim=1)

        output_img_latents = self.sample_images(encoder_hidden_states, encoder_mask, scheduler)
        output_img = self.decode_latents(output_img_latents, return_tensor=False)

        return output_img

    def sample_images(
        self,
        encoder_hidden_states,
        encoder_mask,
        scheduler,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        **kwargs,
    ):
        
        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype


        img_hidden_states_null = torch.zeros_like(encoder_hidden_states, device=device, dtype=dtype)
        img_hidden_states_input = torch.cat([img_hidden_states_null, encoder_hidden_states], 0)

        batch_size = encoder_hidden_states.shape[0]
        latent_size = self.get_model().dit.config.sample_size
        latent_channels = self.get_model().dit.config.in_channels

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # set step values
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        # Repeat z_latents and conditions for each image per prompt
        img_hidden_states_input = img_hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)

        for t in scheduler.timesteps:
            latent_model_input = latents.repeat(2, 1, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.get_model().dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=img_hidden_states_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                added_cond_kwargs={'resolution': None, 'aspect_ratio': None},
                encoder_attention_mask=encoder_mask.repeat_interleave(num_images_per_prompt, dim=0).repeat(2, 1),
            )

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # samples = self.decode_latents(latents, return_tensor=return_tensor)
        # breakpoint()
        return latents

    def decode_latents(self, latents, normalize=True, return_tensor=False):
        return self.get_model().decode_latents(latents, normalize=normalize, return_tensor=return_tensor)

    def prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        tokenizer: AutoTokenizer,
        do_classifier_free_guidance: bool = False,
    ):
        # pdb.set_trace()
        device = self.get_model().device
        dtype = self.get_model().dtype

        has_image, has_text = False, False
        text_prompt, image_prompt = "", []
        img_processor = self.get_vision_tower().image_processor
        negative_prompt = {}

        for x in inputs:
            if isinstance(x, str):
                has_text = True
                text_prompt += x
            else:
                has_image = True
                text_prompt += DEFAULT_IMAGE_TOKEN
                image_prompt.append(img_processor.preprocess(x, return_tensors='pt')['pixel_values'])
        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.cat(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if has_image and not has_text:
            prompt = self.encode_images(image_prompt)
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    negative_prompt[key] = self.encode_images(negative_image)
                prompt = torch.cat([prompt, negative_prompt[key]], dim=0)
        else:
            prompt = self.generate_image(text=[text_prompt], image=image_prompt, tokenizer=tokenizer)
            if do_classifier_free_guidance:
                key = ""
                if key not in negative_prompt:
                    negative_prompt[key] = self.generate_image(text=[""], tokenizer=tokenizer)
                prompt = torch.cat([prompt, negative_prompt[key]], dim=0)
        
        gen_pooling = self.get_gen_pooling()
        n_query = self.get_n_query()
        num_img, _, c = prompt.shape
        if 'pool2d' in gen_pooling and has_text and not 'early' in gen_pooling:
            stride = int(gen_pooling.split('_')[1])
            sqrt_n = int(n_query**0.5)
            prompt = prompt.permute(0, 2, 1).reshape(num_img, -1, sqrt_n, sqrt_n)
            prompt = F.avg_pool2d(prompt, kernel_size=(stride, stride), stride=stride)
            prompt = prompt.reshape(num_img, c, -1).permute(0,2,1)
        return prompt


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("blip3o_qwen_inference", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForInferenceLM)
