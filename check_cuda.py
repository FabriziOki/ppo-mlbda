import torch

def main():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Device count: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} — {mem_gb:.1f} GB VRAM")
        print(f"\nRecommended device: cuda")
    else:
        print("No CUDA GPU detected — training will run on CPU (expect slower performance)")
        print("\nRecommended device: cpu")

if __name__ == "__main__":
    main()
