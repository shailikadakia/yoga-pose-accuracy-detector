import os
from PIL import Image

input_folder = "convert_images"
output_folder = "training-data/goddess"

os.makedirs(output_folder, exist_ok=True)

start_index = 0

# 2. Convert and save new JPGs with incremented names
counter = start_index
for filename in os.listdir(input_folder):
  img_path = os.path.join(input_folder, filename)
  img = Image.open(img_path).convert("RGB")

  new_name = f"goddess_{counter}.png"
  out_path = os.path.join(output_folder, new_name)

  img.save(out_path, "PNG")
  print(f"Saved {out_path}")

  counter += 1
