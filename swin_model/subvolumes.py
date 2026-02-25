import os
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.filters import threshold_otsu
Image.MAX_IMAGE_PIXELS = None


def extract_patches(
        input_folder,
        output_folder,
        patch_size=(250, 250, 250),
        num_patches=1000,
        base_seed=42,
        initial_foreground_ratio=0.8,
        source_entity=None,
        source_sample=None,
):
    os.makedirs(output_folder, exist_ok=True)

    volume_id = os.path.basename(os.path.normpath(input_folder))
    if source_entity is None:
        source_entity = "Unknown"

    if source_sample is None:
        source_sample = volume_id

    volume_seed = abs(hash(volume_id)) % (2 ** 32)  # Each volume has a unique seed
    rng = np.random.default_rng(base_seed + volume_seed)  # Base seed for reproducibility

    max_stuck_count = 500
    relaxation_factor = 0.8
    current_foreground_ratio = initial_foreground_ratio
    stuck_counter = 0

    # Setup file list and dimensions
    image_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    )
    if not image_files:
        raise RuntimeError("No image files found.")

    for f in image_files:
        try:
            img = Image.open(os.path.join(input_folder, f))
            img.load()
        except Exception as e:
            print(f"Error loading {f}: {e}")

    first_img = Image.open(os.path.join(input_folder, image_files[0]))
    W, H = first_img.size
    D = len(image_files)
    pH, pW, pD = patch_size

    if H < pH or W < pW or D < pD:
        raise ValueError(
            f"Volume ({H}x{W}x{D}) is smaller than patch size {patch_size}"
        )

    print(f"Processing volume '{volume_id}' with {D} slices")

    # Foreground threshold calculation (memory-safe)
    num_slices_for_otsu = min(10, D)
    pixels_per_slice = 200_000
    sample_indices = rng.choice(D, size=num_slices_for_otsu, replace=False)
    sample_pixels = []
    for i in sample_indices:
        img = np.array(Image.open(os.path.join(input_folder, image_files[i])), dtype=np.float32)
        flat = img.ravel()
        if flat.size > pixels_per_slice:
            flat = rng.choice(flat, size=pixels_per_slice, replace=False)

        sample_pixels.append(flat)

    sample_pixels = np.concatenate(sample_pixels)
    global_min = float(sample_pixels.min())
    global_max = float(sample_pixels.max())
    threshold = threshold_otsu((sample_pixels - global_min) / (global_max - global_min + 1e-8))
    print(f"Otsu threshold: {threshold:.4f}")

    # Extract subvolumes from full volume
    extracted = 0
    total_attempts = 0
    pbar = tqdm(total=num_patches, desc="Extracting patches")
    while extracted < num_patches:
        h_start = rng.integers(0, H - pH + 1)
        w_start = rng.integers(0, W - pW + 1)
        d_start = rng.integers(0, D - pD + 1)
        # Fast foreground check (middle slice of selected subvolume)
        mid_z = d_start + pD // 2
        mid_slice = np.array(
            Image.open(os.path.join(input_folder, image_files[mid_z])),
            dtype=np.float32,
        )
        mid_crop = mid_slice[h_start:h_start + pH, w_start:w_start + pW]
        fg_ratio = (((mid_crop - global_min) / (global_max - global_min + 1e-8)) > threshold).mean()
        if fg_ratio < current_foreground_ratio:  # If fg of extracted subvolume is not enough, try again
            stuck_counter += 1
            total_attempts += 1
            if stuck_counter > max_stuck_count:  # After too many tries, relax fg requirement
                current_foreground_ratio = max(0.01, current_foreground_ratio * relaxation_factor)
                pbar.write(f"Relaxing foreground ratio to {current_foreground_ratio:.3f}")
                stuck_counter = 0
            continue

        stuck_counter = 0
        # When a good subvolume is found, load it fully and save it
        patch_stack = []
        for i in range(d_start, d_start + pD):
            img = Image.open(os.path.join(input_folder, image_files[i]))
            patch_stack.append(np.array(img.crop((w_start, h_start, w_start + pW, h_start + pH)), dtype=np.float32,))

        patch = np.stack(patch_stack, axis=-1)
        nib.save(nib.Nifti1Image(patch, np.eye(4)), os.path.join(output_folder, f"patch_{extracted:04d}.nii.gz"),)
        extracted += 1
        pbar.update(1)

    pbar.close()

    # Save metadata for reproducibility and tracking
    metadata = {
        # Info about the entity and sample of the volume
        "source_entity": source_entity,
        "source_sample": source_sample,
        "volume_id": volume_id,
        # Patch extraction details
        "patch_size": patch_size,
        "num_patches": num_patches,
        # Reproducibility
        "base_seed": base_seed,
        "volume_seed": int(volume_seed),
        # Foreground sampling
        "initial_foreground_ratio": initial_foreground_ratio,
        "final_foreground_ratio": current_foreground_ratio,
        "otsu_threshold": float(threshold),
    }

    with open(os.path.join(output_folder, "extraction_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Done. Attempts: {total_attempts}. "
        f"Final fg ratio: {current_foreground_ratio:.3f}"
    )


if __name__ == "__main__":
    input_folder = os.path.join(os.getcwd(), "TrainingData", r"data_selma_unused\Nuclei\sample3")
    output_folder = os.path.join(os.getcwd(), "TrainingData", r"data_selma_unused\Nuclei\patches\sample3")
    extract_patches(
        input_folder=input_folder,
        output_folder=output_folder,
        patch_size=(250, 250, 250),
        num_patches=250,
        base_seed=42,
        initial_foreground_ratio=0.8,
        source_entity="Nuclei",
        source_sample="sample3"
    )
