import os
from PIL import Image
from pathlib import Path

def add_new_file(input_folder, output_folder) :
    os.makedirs(output_folder, exist_ok=True)
    path = Path(output_folder)
    label = path.name

    # Find the highest existing index in the output folder
    existing_files = [f for f in os.listdir(output_folder) if f.startswith(label) and f.endswith(".png")]
    existing_indices = []

    for f in existing_files:
        try:
            num = int(f.replace(label + "_", "").replace(".png", ""))
            existing_indices.append(num)
        except ValueError:
            continue

    start_index = max(existing_indices, default=0) + 1

    # Convert and save new JPGs with incremented names
    counter = start_index
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")

        new_name = f"P{label}_{counter}.png"
        out_path = os.path.join(output_folder, new_name)

        img.save(out_path, "PNG")
        print(f"Saved {out_path}")

        counter += 1




