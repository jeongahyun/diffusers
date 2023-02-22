import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os


MODEL_PATH = "/home/data/etc_data/dreambooth/results/pine_trial_7/weights/2000"
PROMPT = "photo of zwx justin bieber bottom view" #@param {type:"string"}
NEGATIVE_PROMPT = "" #@param {type:"string"}
NUM_SAMPLES = 10 #@param {type:"number"}
GUIDANCE_SCALE = 8 #@param {type:"number"}
NUM_INFERENCE_STEPS = 100 #@param {type:"number"}
HEIGHT = 512 #@param {type:"number"}
WIDTH = 512 #@param {type:"number"}
OUTPUT_DIR = f"/home/data/etc_data/dreambooth/results/pine_trial_7"

pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
g_cuda = None
g_cuda = torch.Generator(device='cuda')
seed = 5156 #@param {type:"number"}
g_cuda.manual_seed(seed)

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        negative_prompt=NEGATIVE_PROMPT,
        num_images_per_prompt=NUM_SAMPLES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=g_cuda
    ).images

for i, img in enumerate(images):
    img.save(f"{OUTPUT_DIR}/{i}.png")