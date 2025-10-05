# data.py
import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config
import cv2
import pydicom
import warnings
import pydicom.config as pdcfg

from pydicom.pixel_data_handlers import pillow_handler
try:
    from pydicom.pixel_data_handlers import pylibjpeg_handler
except Exception:
    pylibjpeg_handler = None
try:
    from pydicom.pixel_data_handlers import gdcm_handler
except Exception:
    gdcm_handler = None

handlers = []
if pylibjpeg_handler is not None: handlers.append(pylibjpeg_handler)  # prefer pylibjpeg
if gdcm_handler is not None:      handlers.append(gdcm_handler)       # then GDCM
handlers.append(pillow_handler)                                       
pdcfg.image_handlers = handlers

warnings.filterwarnings(
    "ignore",
    message=".*'Bits Stored' value .* doesn't match the JPEG 2000 data .*",
    category=UserWarning,
    module="pydicom.pixel_data_handlers.pillow_handler",
)

class SuperResolutionDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        lr_image_size = config.IMAGE_SIZE // 2
        
        self.hr_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_image_size, lr_image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            if path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(path, force=True)

                # If compressed, let pylibjpeg/GDCM handle it
                ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
                if ts and ts.is_compressed:
                    ds.decompress()

                # Apply VOI LUT (better CT windowing)
                try:
                    from pydicom.pixel_data_handlers.util import apply_voi_lut
                    image_array = apply_voi_lut(ds.pixel_array, ds)
                except Exception:
                    image_array = ds.pixel_array

                # Ensure MONOCHROME2 (bright bone)
                if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                    image_array = image_array.max() - image_array

                # Normalize to [0,255] uint8
                image_array = image_array.astype(np.float32)
                m, M = image_array.min(), image_array.max()
                image_array = (image_array - m) / (M - m + 1e-6) * 255.0
                image_array = image_array.astype(np.uint8)

                image = Image.fromarray(image_array).convert("L")
            else:
                image = Image.open(path).convert("L")

            # --- NEW: APPLY CLAHE (Grayscale version) ---
            if config.DATA_USE_CLAHE:
                img_np = np.array(image)  # Should be (H, W), grayscale
                clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
                img_clahe = clahe.apply(img_np)
                image = Image.fromarray(img_clahe)
            # --- END OF CLAHE SECTION ---

            # Create HR image for training
            hr_image = self.hr_transform(image)
            # Create LR image for visualization. We create it from the original PIL image.
            lr_image = self.lr_transform(image)
            return {"lr": lr_image, "hr": hr_image}
        except Exception as e:
            print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
            lr_image_size = config.IMAGE_SIZE // 2
            return {"lr": torch.zeros(config.IMAGE_CHANNELS, lr_image_size, lr_image_size),
                    "hr": torch.zeros(config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)}

def get_dataloaders(client_id):
    dataset_info = config.CLIENT_DATASETS[client_id]

    # Allow single str or list of extensions
    exts = dataset_info["ext"]
    if isinstance(exts, str):
        exts = [exts]

    # Build case-insensitive patterns
    patterns = []
    for ext in exts:
        patterns.append(os.path.join(dataset_info["path"], f"**/*.{ext}"))
        patterns.append(os.path.join(dataset_info["path"], f"**/*.{ext.upper()}"))

    # Collect files
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat, recursive=True))

    # Deduplicate + sort
    all_files = sorted(set(all_files))

    if not all_files:
        raise RuntimeError(
            f"No images found for client {client_id} at path {dataset_info['path']} "
            f"with extensions {exts}."
        )

    # Shuffle (stable seed)
    np.random.seed(42)
    np.random.shuffle(all_files)

    # Optional cap per client
    limit = getattr(config, "NUM_IMAGES_PER_CLIENT", None)
    if isinstance(limit, int) and limit > 0:
        files = all_files[:min(limit, len(all_files))]
        if len(all_files) < limit:
            print(f"Warning: Client {client_id} has only {len(all_files)} images, "
                  f"less than requested {limit}. Using all available.")
    else:
        files = all_files

    # Split
    n = len(files)
    train_end = int(n * config.DATA_SPLIT["train"])
    val_end = train_end + int(n * config.DATA_SPLIT["val"])

    train_files = files[:train_end]
    val_files   = files[train_end:val_end]
    test_files  = files[val_end:]

    print(f"Client {client_id}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images.")

    # Datasets
    train_dataset = SuperResolutionDataset(train_files)
    val_dataset   = SuperResolutionDataset(val_files)
    test_dataset  = SuperResolutionDataset(test_files)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


