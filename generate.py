
import datetime, os, json

from diffusers import DiffusionPipeline
import torch

prompt_dict = {
    'pos': 'elf swashbuckler with pointy ears holding a glowing rapier doing a backflip from a parapet walk',
    'neg': 'hat, cap, headwear, head covering' # Set to 'None' if not needed
}

# TODO I could make these argumnets to the script (via 'arparse')
n_runs = 5
n_steps = 40
high_noise_frac = 0.8
n_imgs = 5 # For my system 5 seems a sweetspot (usage of model offload instead of sequential)

# Not sure if there is benifit from the following
# https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16#use-tf32-instead-of-fp32-on-ampere-and-later-cuda-devices
torch.backends.cuda.matmul.allow_tf32 = True

gen_start = datetime.datetime.now()
gen_start_str = gen_start.strftime("%Y-%m-%d_%H-%M-%S")
print(f'Started at {gen_start_str}')

# Create output folder, if needed
outdir = f'generated/{gen_start_str}'
# Should not exist, but just to be sure:
if not os.path.isdir(outdir): 
    os.makedirs(outdir, exist_ok = True)

# Save prompt
with open(f'{outdir}/prompt.json', mode='w') as jfile:
    json.dump(prompt_dict, fp=jfile)

# Run generation
for run_i in range(n_runs):
    print(f'\n# Run {run_i + 1} of {n_runs}')

    print(f'Generating {n_imgs} latent images')
    base = DiffusionPipeline.from_pretrained(
        './stable-diffusion-xl-base-1.0', 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant='fp16'
    )
    # model offload is quicker than sequential, however, execds my memory for more than 5 images
    if n_imgs <= 5:
        base.enable_model_cpu_offload()
    else: 
        base.enable_sequential_cpu_offload()
    latent_images  = base(
        prompt = prompt_dict['pos'], 
        negative_prompt = prompt_dict['neg'],
        num_images_per_prompt = n_imgs,
        num_inference_steps = n_steps,
        denoising_end = high_noise_frac,
        output_type = "latent"
    ).images
    del base # Make sure memory gets freed TODO check for better way

    print(f'Generating {n_imgs} refined images from laten ones')
    refiner = DiffusionPipeline.from_pretrained(
        './stable-diffusion-xl-refiner-1.0', 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant='fp16'
    )
    if n_imgs <= 5:
        refiner.enable_model_cpu_offload()
    else: 
        refiner.enable_sequential_cpu_offload()
    # Otherwise VAE stage will run out of memory (might not be necessary for n_imgs > 5)
    if n_imgs > 2:    
        refiner.enable_vae_slicing()
    final_images = refiner(
        prompt = prompt_dict['pos'], 
        negative_prompt = prompt_dict['neg'],
        num_images_per_prompt = n_imgs,
        num_inference_steps = n_steps,
        denoising_start = high_noise_frac,
        image = latent_images
    ).images
    del refiner # Make sure memory gets freed TODO check for better way

    # I could also do all images at once, however, my GPU memory is not large enough...
    for img_i, fimg in enumerate(final_images):
        img_file = f'{outdir}/{run_i}-{img_i}.png'
        # print(f'Saving final image {img_i} to {img_file}')
        fimg.save(img_file)

print(f'\nGeneration of {n_runs*n_imgs} images took {datetime.datetime.now() - gen_start}')