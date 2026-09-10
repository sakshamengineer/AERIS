import torch

def main():
    print("=" * 60)
    print("GPU / CUDA CHECK")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n❌ CUDA GPU is not available.")
        print("Training will fall back to CPU, which is painfully slow for Qwen2.5-VL.")
        return

    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)

        total_memory = gpu.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)

        print(f"\nGPU {i}")
        print("-" * 40)
        print(f"Name:              {gpu.name}")
        print(f"VRAM:              {total_memory:.2f} GB")
        print(f"Allocated memory:  {allocated:.2f} GB")
        print(f"Reserved memory:   {reserved:.2f} GB")

    print("\n" + "=" * 60)
    print("GPU CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()