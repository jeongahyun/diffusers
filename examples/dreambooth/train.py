PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"
PRETRAINED_VAE_NAME_OR_PATH = "stabilityai/sd-vae-ft-mse"
OUTPUT_DIR = "/home/data/etc_data/dreambooth/results/pine_trial_7/weights"
REVISION = "fp16"
PRIOR_LOSS_WEIGHT = 1.0
SEED = 1337
RESOLUTION = 512
TRAIN_BATCH_SIZE = 1
MIXED_PRECISION = "fp16"
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-6
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 0
NUM_CLASS_IMAGES = 1000
SAMPLE_BATCH_SIZE = 4
MAX_TRAIN_STEPS = 2000
SAVE_INTERVAL = 1000
SAVE_SAMPLE_PROMPT = "photo of zwx justin bieber"
CONCEPTS_LIST = "/home/data/etc_data/dreambooth/results/pine_trial_7/concepts_list.json"
INSTANCE_PROMPT = "photo of zwx justin bieber"
CLASS_PROMPT = "photo of a justin bieber"
INSTANCE_DATA_DIR = "/home/data/etc_data/dreambooth/zwx"
CLASS_DATA_DIR = "/home/data/etc_data/dreambooth/justing bieber"  # regularization images dir


import json
import os
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print(f"[*] Weights will be saved at {OUTPUT_DIR}")
cmd = f"mkdir -p {OUTPUT_DIR}"
os.system(cmd)

concepts_list = [
    {
        "instance_prompt":      f"{INSTANCE_PROMPT}",
        "class_prompt":         f"{CLASS_PROMPT}",
        "instance_data_dir":    f"{INSTANCE_DATA_DIR}",
        "class_data_dir":       f"{CLASS_DATA_DIR}"
    },
]

with open(f"{CONCEPTS_LIST}", "w") as f:
    json.dump(concepts_list, f, indent=4)

cmd = f"accelerate launch train_dreambooth.py \
--pretrained_model_name_or_path='{PRETRAINED_MODEL_NAME_OR_PATH}' \
--pretrained_vae_name_or_path='{PRETRAINED_VAE_NAME_OR_PATH}' \
--output_dir='{OUTPUT_DIR}' \
--revision='{REVISION}' \
--with_prior_preservation --prior_loss_weight={PRIOR_LOSS_WEIGHT} \
--seed={SEED} \
--resolution={RESOLUTION} \
--train_batch_size={TRAIN_BATCH_SIZE} \
--train_text_encoder \
--mixed_precision='{MIXED_PRECISION}' \
--gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS} \
--learning_rate={LEARNING_RATE} \
--lr_scheduler='{LR_SCHEDULER}' \
--lr_warmup_steps={LR_WARMUP_STEPS} \
--num_class_images={NUM_CLASS_IMAGES} \
--sample_batch_size={SAMPLE_BATCH_SIZE} \
--max_train_steps={MAX_TRAIN_STEPS} \
--save_interval={SAVE_INTERVAL} \
--save_sample_prompt='{SAVE_SAMPLE_PROMPT}' \
--concepts_list='{CONCEPTS_LIST}'"
print(cmd)

os.system(cmd)

# preview 확인
WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

weights_folder = OUTPUT_DIR
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key=lambda x: int(x))

row = len(folders)
col = len(os.listdir(os.path.join(weights_folder, folders[0], "samples")))
scale = 4
fig, axes = plt.subplots(row, col, figsize=(col*scale, row*scale), gridspec_kw={'hspace': 0, 'wspace': 0})

for i, folder in enumerate(folders):
    folder_path = os.path.join(weights_folder, folder)
    image_folder = os.path.join(folder_path, "samples")
    images = [f for f in os.listdir(image_folder)]
    for j, image in enumerate(images):
        if row == 1:
            currAxes = axes[j]
        else:
            currAxes = axes[i, j]
        if i == 0:
            currAxes.set_title(f"Image {j}")
        if j == 0:
            currAxes.text(-0.1, 0.5, folder, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
        image_path = os.path.join(image_folder, image)
        img = mpimg.imread(image_path)
        currAxes.imshow(img, cmap='gray')
        currAxes.axis('off')
        
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/preview.png', dpi=72)

# ckpt 파일 변환
ckpt_path = WEIGHTS_DIR + "/model.ckpt"
half_arg = ""
fp16 = True
if fp16:
    half_arg = "--half"

cmd = f"python convert_diffusers_to_original_stable_diffusion.py \
--model_path {WEIGHTS_DIR} \
--checkpoint_path {ckpt_path} \
{half_arg}"

os.system(cmd)
print(f"[*] Converted ckpt saved at {ckpt_path}")