import os


OUTPUT_FOLDER = 'C:/Users/Admin/Desktop/SHRG-Net/Data/test1_image/train'
F = "C:/Users/Admin/Desktop/SHRG-Net/Data/images/InterHand2.6M_5fps_batch1/images/test/Capture1/ROM01_No_Interaction_2_Hand"
VAL_FOLDER = "C:/Users/Admin/Desktop/SHRG-Net/Data/test1_image/test"


import os
import random
import shutil
from pathlib import Path

def flatten_and_split(source_dir, train_dir, val_dir, split_ratio=0.7, seed=42):
    source_path = Path(source_dir).resolve()
    train_path = Path(train_dir).resolve()
    val_path = Path(val_dir).resolve()
    
    # Create main destination folders
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # Collect all images from all nested subfolders into a single flat list
    all_images = []
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)
        for f in files:
            if Path(f).suffix.lower() in valid_extensions:
                all_images.append(root_path / f)
                
    if not all_images:
        print(f"Error: No valid images found in '{source_path}' or its subfolders.")
        return
        
    print(f"Found {len(all_images)} total images across all subfolders.")
    
    # Shuffle randomly for an unbiased split
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_images)
    
    # Split 70/30
    split_index = int(len(all_images) * split_ratio)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]
    
    def copy_flat(image_list, dest_folder):
        for img_path in image_list:
            dest_file = dest_folder / img_path.name
            
            # If a file with the same name already exists from a different subfolder,
            # prefix it with its parent folder name to prevent overwriting
            if dest_file.exists():
                parent_name = img_path.parent.name
                dest_file = dest_folder / f"{parent_name}_{img_path.name}"
                
            shutil.copy(img_path, dest_file)

    # Copy to train folder
    print(f"Copying {len(train_images)} images to '{train_path}'...")
    copy_flat(train_images, train_path)
    
    # Copy to validation folder
    print(f"Copying {len(val_images)} images to '{val_path}'...")
    copy_flat(val_images, val_path)
    
    print("\nDone! All images have been flattened and split successfully.")


if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    SOURCE_FOLDER = F
    TRAIN_FOLDER  = OUTPUT_FOLDER
    VAL_FOLDER    = VAL_FOLDER
    
    flatten_and_split(SOURCE_FOLDER, TRAIN_FOLDER, VAL_FOLDER, split_ratio=0.7)