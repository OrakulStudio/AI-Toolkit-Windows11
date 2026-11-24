import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

from toolkit.prompt_utils import PromptEmbeds
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.dequantize import patch_dequantization_on_save

from .src.transformer import Flux3Transformer, Flux3TransformerParams
from .src.image_vae import Flux3ImageVAE
from .src.pipeline import (
    Flux3Pipeline,
    build_text_conditioning,
    pack_image_latents,
    pack_i2i_image_condition,
)
from .src.text_encoder import encode_flux3_prompt

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
}

QWEN3_VL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_REPO = "black-forest-labs/flux-3-image-klein-base"


class Flux3Model(BaseModel):
    arch = "flux3"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Flux3Transformer"]
        self.use_raw_control_images = True
        self.has_multiple_control_images = True
        self.use_old_lokr_format = False
        self.latent_space_version = "flux3_v1"
        self.match_target_res = bool(
            self.model_config.model_kwargs.get("match_target_res", True)
        )

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(shift=6.93, **scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_te(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen3-VL-4B")

        text_encoder = AutoModelForVision2Seq.from_pretrained(
            QWEN3_VL_PATH,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
        )

        text_encoder.to(self.device_torch, dtype=dtype)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3-VL")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            from optimum.quanto import freeze
            freeze(text_encoder)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)
        processor.tokenizer.padding_side = "right"
        return text_encoder, processor

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading FLUX.3 Image Klein")

        model_path = self.model_config.name_or_path or DEFAULT_REPO

        # ====================== TRANSFORMER ======================
        self.print_and_status_update("Loading transformer")

        with torch.device("meta"):
            transformer = Flux3Transformer(Flux3TransformerParams())
            # --- СЮДА ЕЁ ПЕРЕНЕСИ ---
            transformer.config = type('obj', (object,), {'patch_size': 2})()
            # -----------------------

        transformer_path = model_path
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, "model.safetensors")
            if os.path.exists(candidate):
                transformer_path = candidate

        if not os.path.isfile(transformer_path):
            from huggingface_hub import hf_hub_download
            transformer_path = hf_hub_download(
                repo_id=model_path if not os.path.isdir(model_path) else DEFAULT_REPO,
                filename="model.safetensors",
                token=os.getenv("HF_TOKEN", None),
            )

        self.print_and_status_update(f"Loading weights from {transformer_path}")
        state_dict = load_file(transformer_path, device="cpu")
        state_dict = Flux3Transformer.convert_state_dict_on_load(state_dict)

        for k, v in state_dict.items():
            state_dict[k] = v.to(dtype)

        transformer.load_state_dict(state_dict, assign=True)
        del state_dict
        flush()

        transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            patch_dequantization_on_save(transformer)
            self.print_and_status_update("Quantizing transformer")
            quantize_model(self, transformer)
            flush()
        else:
            transformer.to(self.device_torch, dtype=dtype)

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        # ====================== TEXT ENCODER ======================
        text_encoder, processor = self.load_te()

        # ====================== VAE ======================
        self.print_and_status_update("Loading VAE")
        vae = Flux3ImageVAE()

        vae_path = None
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, "ae.safetensors")
            if os.path.exists(candidate):
                vae_path = candidate

        if vae_path is None:
            from huggingface_hub import hf_hub_download
            vae_path = hf_hub_download(
                repo_id=model_path if not os.path.isdir(model_path) else DEFAULT_REPO,
                filename="ae.safetensors",
                token=os.getenv("HF_TOKEN", None),
            )

        vae_sd = load_file(vae_path, device="cpu")
        vae.load_state_dict(vae_sd, strict=False)
        vae.to(self.device_torch, dtype=dtype)
        vae.eval().requires_grad_(False)
        del vae_sd
        flush()

        self.noise_scheduler = self.get_train_scheduler()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.model = transformer
        self.pipeline = Flux3Pipeline(self)

        self.print_and_status_update("FLUX.3 Image Klein loaded")


    def get_prompt_embeds(self, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]

        if next(self.text_encoder.parameters()).device != self.device_torch:
            self.text_encoder.to(self.device_torch)

        ctx = encode_flux3_prompt(
            self.text_encoder,
            self.processor,
            prompt,
            max_length=8192,
            dtype=self.torch_dtype,
            pad_multiple=80,
        )
        return PromptEmbeds(ctx)

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        device = self.device_torch
        dtype = self.torch_dtype

        if next(self.model.parameters()).device == torch.device("cpu"):
            self.model.to(device)

        batch_size = latent_model_input.shape[0]

        with torch.no_grad():
            t = timestep.to(device, dtype=torch.float32) / 1000.0
            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.shape[0] != batch_size:
                t = t.expand(batch_size)

            raw_embeds = text_embeddings.text_embeds if hasattr(text_embeddings, "text_embeds") else text_embeddings
            
            # Режем 3D-тензор батча на список 2D-тензоров, как того хочет FLUX
            if torch.is_tensor(raw_embeds) and raw_embeds.ndim == 3:
                embeds = [raw_embeds[i] for i in range(raw_embeds.shape[0])]
            else:
                embeds = raw_embeds if isinstance(raw_embeds, list) else [raw_embeds]

            # ВОТ ЭТУ СТРОКУ Я ПРОЕБАЛ:
            ctx, ctx_ids, ts_ctx = build_text_conditioning(embeds, device, dtype)
            

            latent_model_input = latent_model_input.to(device, dtype)
            _, _, h_lat, w_lat = latent_model_input.shape
            tokens, ids = pack_image_latents(latent_model_input)

            model_kwargs = {
                "x_image": tokens.to(dtype),
                "x_image_ids": ids,
                "x_image_timesteps": t.to(dtype)[:, None]
                * torch.ones(tokens.shape[:2], device=device, dtype=dtype),
            }

            control_list = None
            if batch is not None:
                if getattr(batch, "control_tensor_list", None) is not None:
                    control_list = batch.control_tensor_list
                elif getattr(batch, "control_tensor", None) is not None:
                    control_list = [
                        list(ct) if ct.ndim == 4 else [ct]
                        for ct in batch.control_tensor
                    ]

            if control_list is not None:
                target_pixels = h_lat * w_lat * 16 * 16
                ref_latents = []
                for controls in control_list:
                    sample_refs = []
                    for ctrl in controls:
                        if ctrl.ndim == 3:
                            ctrl = ctrl.unsqueeze(0)
                        h, w = ctrl.shape[-2:]
                        if self.match_target_res or (h * w > target_pixels):
                            ratio = h / max(w, 1)
                            new_h = math.sqrt(target_pixels * ratio)
                            new_w = new_h / ratio
                            new_h = max(16, int(round(new_h / 16)) * 16)
                            new_w = max(16, int(round(new_w / 16)) * 16)
                            ctrl = F.interpolate(
                                ctrl.float(), size=(new_h, new_w), mode="bilinear"
                            )
                        lat = self.encode_images(ctrl * 2 - 1, device=device, dtype=dtype)
                        sample_refs.append(lat[0])
                    ref_latents.append(sample_refs)

                xc, xc_ids, xc_ts = pack_i2i_image_condition(ref_latents, device, dtype)
                model_kwargs["x_image_cond"] = xc
                model_kwargs["x_image_cond_ids"] = xc_ids
                model_kwargs["x_image_cond_timesteps"] = xc_ts

        pred = self.model(
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps_ctx=ts_ctx,
            **model_kwargs,
        )
        out = pred["x_image"]
        return out.reshape(batch_size, h_lat, w_lat, -1).permute(0, 3, 1, 2)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        if device is None:
            device = self.device_torch
        if dtype is None:
            dtype = self.torch_dtype

        if next(self.vae.parameters()).device == torch.device("cpu"):
            self.vae.to(device)

        images = torch.stack(
            [img.to(device, dtype=self.vae.dtype) for img in image_list]
        )
        latents = self.vae.encode(images)
        return latents.to(device, dtype=dtype)

    def get_base_model_version(self):
        return "flux3"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return [
            "content_mode_blocks.image",
            "content_mode_blocks.image_cond",
            "txt_mode_blocks",
            "single_blocks",
        ]

    def convert_lora_weights_before_save(self, state_dict):
        return {
            k.replace("transformer.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    def convert_lora_weights_before_load(self, state_dict):
        return {
            k.replace("diffusion_model.", "transformer."): v
            for k, v in state_dict.items()
        }