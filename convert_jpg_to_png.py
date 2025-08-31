import os
from PIL import Image

input_folder = "convert_images"
output_folder = "training-data/goddess"

os.makedirs(output_folder, exist_ok=True)

# 1. Find the highest existing index in the output folder
existing_files = [f for f in os.listdir(output_folder) if f.startswith("goddess") and f.endswith(".png")]
existing_indices = []

for f in existing_files:
    try:
        num = int(f.replace("downward_dog_", "").replace(".png", ""))
        existing_indices.append(num)
    except ValueError:
        continue

start_index = max(existing_indices, default=0) + 1

# 2. Convert and save new JPGs with incremented names
counter = start_index
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")

        new_name = f"goddess{counter}.png"
        out_path = os.path.join(output_folder, new_name)

        img.save(out_path, "PNG")
        print(f"Saved {out_path}")

        counter += 1
