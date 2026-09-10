from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def find_identifier(root: Path,identifier: str,max_results: int = 10):

    results = []

    print(f"\nSearching for:\n{identifier}")
    for path in root.rglob("*"):
        if identifier.lower() in path.name.lower():
            results.append(path)

            print(f"FOUND: {path}")
            if len(results) >= max_results:
                break
    if not results:
        print("No matching files/directories found.")
    return results

def inspect_data_directories():

    print("=" * 60)
    print("SATQUERY AI - BIGEARTHNET IMAGE CHECK")
    print("=" * 60)

    print(f"\nProject root:\n{PROJECT_ROOT}")
    print("\nChecking data directory...")
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        print("\nNo data directory exists yet:")
        print(data_dir)
        return

    print(f"\nData directory found:")
    print(data_dir)
    print("\nTop-level contents:")
    for item in data_dir.iterdir():
        if item.is_dir():
            print(f"[DIR]  {item.name}")
        else:
            print(f"[FILE] {item.name}")

def main():

    inspect_data_directories()
    data_dir = PROJECT_ROOT / "data"

    if not data_dir.exists():
        return

    sentinel_2_patch = ("S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_37_88")
    sentinel_1_patch = ("S1B_IW_GRDH_1SDV_20170612T165809_33UUP_37_88")

    print("\n" + "=" * 60)
    print("SEARCHING FOR SENTINEL-2 PATCH")
    print("=" * 60)

    find_identifier(root=data_dir,identifier=sentinel_2_patch)

    print("\n" + "=" * 60)
    print("SEARCHING FOR SENTINEL-1 PATCH")
    print("=" * 60)

    find_identifier(root=data_dir,identifier=sentinel_1_patch,)

    print("\n" + "=" * 60)
    print("BIGEARTHNET DIRECTORY CHECK")
    print("=" * 60)

    possible_names = [
        "BigEarthNet",
        "BigEarthNet-v2",
        "BigEarthNet-v2.0",
        "BigEarthNet-S1",
        "BigEarthNet-S2",
        "bigearthnet",
        "bigearthnet-v2",
    ]

    for name in possible_names:
        candidate = PROJECT_ROOT / name
        if candidate.exists():
            print(f"FOUND: {candidate}")

if __name__ == "__main__":
    main()