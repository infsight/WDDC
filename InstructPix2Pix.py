import PIL
# import requests
import torch
# from io import BytesIO

from diffusers import StableDiffusionInstructPix2PixPipeline
import os
from tqdm import tqdm


# def download_image(url):
#     response = requests.get(url)
#     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


# img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

# image = download_image(img_url).resize((512, 512))

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# prompt = "Generating sunrise style scenes"
# image = PIL.Image.open('autodl-tmp/diffusion/water_0001.jpg').convert("RGB")
# image = pipe(prompt=prompt, image=image).images[0]
# image.save('autodl-tmp/diffusion/water_0001_night.jpg')

styles = ["rainy", "foggy", "dusk", "sunrise", "nighttime", "blurry"]
img_dir = "autodl-tmp/UW-Bench-v2-4/train_supervised/JPEGImages"
img_name_list = sorted(os.listdir(img_dir))
pbar = tqdm(total=len(img_name_list), leave=True, desc='image')
for i in range(len(img_name_list)):
    img_name = img_name_list[i]
    img_path = os.path.join(img_dir, img_name)
    init_image = PIL.Image.open(img_path).convert("RGB")
    
    # j = random.randint(0, 5)
    for j in range(6):
        prompt = f"Generate {styles[j]} style scenes"

        # pass prompt and image to pipeline
        image = pipe(prompt=prompt, image=init_image).images[0]
        aug_img_name = styles[j] + '_' + img_name
        aug_img_path = os.path.join('autodl-tmp/diffusion/inp2p', aug_img_name)
        image.save(aug_img_path)
    if pbar is not None:
        pbar.update(1)

if pbar is not None:
    pbar.close()