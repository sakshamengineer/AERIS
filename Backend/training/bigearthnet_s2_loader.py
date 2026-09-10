import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "training_manifest_large.json"
)

BIGEARTHNET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "BigEarthNet-S2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_band(array):
    """
    Percentile normalization for a Sentinel-2 band.
    Converts the band into 0-255 uint8.
    """

    array = array.astype(np.float32)

    valid = array[np.isfinite(array)]

    if valid.size == 0:
        return np.zeros_like(
            array,
            dtype=np.uint8
        )

    low = np.percentile(valid, 2)
    high = np.percentile(valid, 98)

    if high <= low:
        return np.zeros_like(
            array,
            dtype=np.uint8
        )

    normalized = (
        (array - low)
        / (high - low)
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    return (
        normalized * 255
    ).astype(np.uint8)


# ============================================================
# FIND PATCH DIRECTORY
# ============================================================

def find_patch_directory(patch_id):

    # Search recursively for the exact patch directory.
    matches = list(
        BIGEARTHNET_ROOT.rglob(patch_id)
    )

    if not matches:
        return None

    for match in matches:

        if match.is_dir():
            return match

    return None


# ============================================================
# FIND BAND
# ============================================================

def find_band(patch_dir, band_name):

    matches = list(
        patch_dir.glob(f"*_{band_name}.tif")
    )

    if not matches:
        matches = list(
            patch_dir.glob(f"*_{band_name}.jp2")
        )

    if not matches:
        return None

    return matches[0]


# ============================================================
# PROCESS ONE PATCH
# ============================================================

def process_patch(patch_id):

    output_path = (
        OUTPUT_DIR
        / f"{patch_id}_rgb.png"
    )

    # Skip already processed patches.
    if output_path.exists():

        print(
            f"[SKIP] {patch_id}"
        )

        return True

    patch_dir = find_patch_directory(
        patch_id
    )

    if patch_dir is None:

        print(
            f"[MISSING PATCH] {patch_id}"
        )

        return False

    print(
        f"\n[PROCESSING] {patch_id}"
    )

    print(
        f"Directory: {patch_dir}"
    )

    # --------------------------------------------------------
    # Sentinel-2 RGB bands
    #
    # B04 = Red
    # B03 = Green
    # B02 = Blue
    # --------------------------------------------------------

    band_paths = {
        "B04": find_band(
            patch_dir,
            "B04"
        ),
        "B03": find_band(
            patch_dir,
            "B03"
        ),
        "B02": find_band(
            patch_dir,
            "B02"
        ),
    }

    for band, path in band_paths.items():

        if path is None:

            print(
                f"[MISSING BAND] "
                f"{patch_id}: {band}"
            )

            return False

    # --------------------------------------------------------
    # Read bands
    # --------------------------------------------------------

    bands = {}

    for band, path in band_paths.items():

        with rasterio.open(path) as src:

            data = src.read(1)

        bands[band] = normalize_band(
            data
        )

    # --------------------------------------------------------
    # Ensure same spatial dimensions
    # --------------------------------------------------------

    shapes = {
        band: data.shape
        for band, data in bands.items()
    }

    if len(set(shapes.values())) != 1:

        print(
            f"[SHAPE ERROR] {patch_id}"
        )

        print(shapes)

        return False

    # --------------------------------------------------------
    # Stack RGB
    # --------------------------------------------------------

    rgb = np.stack(
        [
            bands["B04"],
            bands["B03"],
            bands["B02"],
        ],
        axis=-1
    )

    image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    image.save(
        output_path
    )

    print(
        f"[SAVED] {output_path}"
    )

    print(
        f"Size: {image.size}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BIGEARTHNET-S2 PATCH PROCESSING")
    print("=" * 60)

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    if not BIGEARTHNET_ROOT.exists():

        raise FileNotFoundError(
            f"BigEarthNet-S2 directory not found:\n"
            f"{BIGEARTHNET_ROOT}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)

    patches = manifest.get(
        "patches",
        []
    )

    print(
        f"\nPatches in manifest: "
        f"{len(patches)}"
    )

    successful = 0
    failed = 0

    for patch in patches:

        patch_id = patch.get(
            "patch_id"
        )

        if not patch_id:
            continue

        result = process_patch(
            patch_id
        )

        if result:
            successful += 1
        else:
            failed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()