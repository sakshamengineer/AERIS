from pathlib import Path
import json
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

BIGEARTHNET_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
)

MANIFEST_PATH = (
    BIGEARTHNET_DIR
    / "training_manifest.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Only inspect the first few levels initially.
# We do NOT want to recursively print millions of files.

MAX_DEPTH = 3

# Number of files/directories to display per directory.
MAX_ITEMS_PER_DIRECTORY = 20


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        print(
            f"ERROR: Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

        return None

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# SHOW DIRECTORY STRUCTURE
# ============================================================

def show_structure(
    directory: Path,
    depth: int = 0,
):

    if depth > MAX_DEPTH:
        return

    indent = "    " * depth

    try:
        items = sorted(
            directory.iterdir(),
            key=lambda x: (
                not x.is_dir(),
                x.name.lower(),
            ),
        )

    except PermissionError:

        print(
            f"{indent}[NO PERMISSION] "
            f"{directory.name}"
        )

        return

    except OSError as error:

        print(
            f"{indent}[ERROR] "
            f"{directory.name}: {error}"
        )

        return

    if not items:
        print(
            f"{indent}[EMPTY] "
            f"{directory.name}"
        )

        return

    for item in items[
        :MAX_ITEMS_PER_DIRECTORY
    ]:

        if item.is_dir():

            print(
                f"{indent}[DIR]  "
                f"{item.name}"
            )

            show_structure(
                item,
                depth + 1,
            )

        else:

            try:
                size_gb = (
                    item.stat().st_size
                    / (1024 ** 3)
                )

                print(
                    f"{indent}[FILE] "
                    f"{item.name} "
                    f"({size_gb:.3f} GB)"
                )

            except OSError:

                print(
                    f"{indent}[FILE] "
                    f"{item.name}"
                )

    if len(items) > MAX_ITEMS_PER_DIRECTORY:

        print(
            f"{indent}... "
            f"{len(items) - MAX_ITEMS_PER_DIRECTORY} "
            f"more items"
        )


# ============================================================
# SEARCH FOR ONE PATCH
# ============================================================

def search_patch(
    root: Path,
    patch_id: str,
):

    print(
        "\n" + "=" * 60
    )

    print(
        "SEARCHING FOR S2 PATCH"
    )

    print(
        "=" * 60
    )

    print(
        f"\nPatch ID:\n{patch_id}"
    )

    print(
        "\nSearching..."
    )

    matches = []

    # Search only by the complete patch identifier.
    #
    # This may take some time because the dataset is large,
    # but we only perform this for ONE patch.

    for path in root.rglob("*"):

        if patch_id.lower() in path.name.lower():

            matches.append(path)

            print(
                f"\nFOUND:\n{path}"
            )

            # One exact match is enough for now.
            if len(matches) >= 5:
                break

    if not matches:

        print(
            "\nNo exact patch-name match found."
        )

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SATQUERY AI - BIGEARTHNET S2 INSPECTOR")
    print("=" * 60)

    print(
        f"\nBigEarthNet directory:"
    )

    print(
        BIGEARTHNET_DIR
    )

    # --------------------------------------------------------
    # CHECK DIRECTORY
    # --------------------------------------------------------

    if not BIGEARTHNET_DIR.exists():

        print(
            "\nERROR:"
        )

        print(
            "BigEarthNet directory does not exist."
        )

        print(
            "\nExpected:"
        )

        print(
            BIGEARTHNET_DIR
        )

        return

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = load_manifest()

    if manifest is None:
        return

    print(
        "\nManifest information:"
    )

    print(
        f"Annotations: "
        f"{manifest.get('annotation_count')}"
    )

    print(
        f"Unique patches: "
        f"{manifest.get('unique_patch_count')}"
    )

    # --------------------------------------------------------
    # DIRECTORY STRUCTURE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BIGEARTHNET S2 DIRECTORY STRUCTURE"
    )

    print(
        "=" * 60
    )

    show_structure(
        BIGEARTHNET_DIR
    )

    # --------------------------------------------------------
    # GET FIRST PATCH
    # --------------------------------------------------------

    patches = manifest.get(
        "patches",
        [],
    )

    if not patches:

        print(
            "\nERROR: Manifest contains no patches."
        )

        return

    first_patch = patches[0]

    patch_id = first_patch.get(
        "patch_id"
    )

    if not patch_id:

        print(
            "\nERROR: First patch has no patch_id."
        )

        return

    # --------------------------------------------------------
    # SEARCH FIRST PATCH
    # --------------------------------------------------------

    search_patch(
        root=BIGEARTHNET_DIR,
        patch_id=patch_id,
    )
    
    print(
        "\n" + "=" * 60
    )

    print(
        "INSPECTION COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()