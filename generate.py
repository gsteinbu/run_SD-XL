
import datetime, os

from PIL import Image
from diffusers import DiffusionPipeline
import torch

pos_prompt = 'blind fantasy dwarf with colorful hair and a blindfold'
neg_prompt = 'headgear, hat, cap, headband'

# Not sure if there is benifit from the following
# https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16#use-tf32-instead-of-fp32-on-ampere-and-later-cuda-devices
torch.backends.cuda.matmul.allow_tf32 = True

base = DiffusionPipeline.from_pretrained(
    './stable-diffusion-xl-base-1.0', 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant='fp16'
)
base.enable_model_cpu_offload()
refiner = DiffusionPipeline.from_pretrained(
    './stable-diffusion-xl-refiner-1.0', 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant='fp16'
)
refiner.enable_model_cpu_offload()

n_steps = 50
high_noise_frac = 0.8
n_imgs = 4

gen_start = datetime.datetime.now()
gen_start_str = gen_start.strftime("%Y-%m-%d_%H-%M-%S")
print(f'Started at {gen_start_str}')
print(f'Generating {n_imgs} latent images (all at once)')
image  = base(
    prompt = pos_prompt, 
    negative_prompt = neg_prompt,
    num_images_per_prompt = n_imgs,
    num_inference_steps = n_steps,
    denoising_end = high_noise_frac,
    output_type="latent"
)
# Explicitly drop the base to CPU, otherwise my GPU memory wont be enough...
base.to('cpu')
print(f'Generating {n_imgs} refined images from laten ones (all seperate)')
# I could also do all images at once, however, my GPU memory is not large enough...
final_images = [
    refiner(
        prompt = pos_prompt, 
        negative_prompt = neg_prompt,
        num_images_per_prompt = 1,
        num_inference_steps = n_steps,
        denoising_start = high_noise_frac,
        image = img
    ).images[0]
    for img in image.images
]
refiner.to('cpu')

print(f'Generation duration: {datetime.datetime.now() - gen_start}')

# Create output folder if needed
if not os.path.isdir('generated'):
    os.makedirs('generated')

img_file = f'generated/{gen_start_str}.png'
print(f'Saving images to {img_file}')
full_image = Image.new('RGB', (n_imgs * final_images[0].width, final_images[0].height))
for i in range(n_imgs):
    full_image.paste(final_images[i], (i * final_images[0].width, 0))
full_image.save(img_file)