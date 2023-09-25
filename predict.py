import os
import subprocess
import time
from typing import List

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler
)

from lora_diffusion import LoRAManager, monkeypatch_remove_lora
from hashlib import sha512
import requests

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)

def url_local_fn(url):
    return sha512(url.encode()).hexdigest() + ".safetensors"

def download_lora(url):
    fn = url_local_fn(url)

    if not os.path.exists(fn):
        print(f"Starting download lora from url {url}")
        # stream chunks of the file to disk
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print("Using disk cache...")

    print(f"------- End download lora {url}")
    return fn

def set_lora(self, urllists: List[str], scales: List[float]):
    assert len(urllists) == len(scales), "Number of LoRAs and scales must match."

    merged_fn = url_local_fn(f"{'-'.join(urllists)}")

    if self.loaded == merged_fn:
        print("The requested LoRAs are loaded.")
        assert self.lora_manager is not None
    else:

        st = time.time()
        self.lora_manager = LoRAManager(
            [download_lora(url) for url in urllists], self.pipe
        )
        self.loaded = merged_fn
        print(f"merging time: {time.time() - st}")

    self.lora_manager.tune(scales)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("setup took: ", time.time() - start)


    @torch.inference_mode()
    def predict(
        self,
        base_model: str = Input(
            description="base_model HF sintax",
            default="SG161222/RealVisXL_V1.0",
        ),
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_urls: str = Input(
            description="List of urls for safetensors of lora models, seperated with | .",
            default="",
        ),
        ora_scales: str = Input(
            description="List of scales for safetensors of lora models, seperated with | ",
            default="0.5",
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if base_model is None:
            raise Exception(
                f"Base Model is required."
            )

        sdxl_kwargs = {}
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height

        print("Loading sdxl pipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        )
        pipe.to("cuda")

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        if len(lora_urls) > 0:
            lora_urls = [u.strip() for u in lora_urls.split("|")]
            lora_scales = [float(s.strip()) for s in lora_scales.split("|")]
            self.set_lora(lora_urls, lora_scales)
            prompt = self.lora_manager.prompt(prompt)
        else:
            print("No LoRA models provided, using default model...")
            monkeypatch_remove_lora(self.pipe.unet)
            monkeypatch_remove_lora(self.pipe.text_encoder)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **sdxl_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/output-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
