"""GPU and CUDA verification script for ForgeLM.

Run with:
    python test_gpu.py
"""

import sys

import torch


def main() -> None:
    """Run GPU diagnostics and report results."""
    print("=" * 50)
    print("ForgeLM -- GPU Verification")
    print("=" * 50)

    # 1. Python version
    print(f"\nPython version : {sys.version}")

    # 2. PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # 3. CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available : {cuda_available}")

    if not cuda_available:
        print("\n[WARNING] CUDA is not available.")
        print("Training will fall back to CPU, which will be significantly slower.")
        print("Ensure NVIDIA drivers and a CUDA-compatible PyTorch build are installed.")
        print("\nInstall PyTorch with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 50)
        return

    # 4. GPU name
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU name       : {gpu_name}")

    # 5. GPU memory
    props = torch.cuda.get_device_properties(0)
    total_mem_mb = props.total_mem / 1024**2
    print(f"GPU memory     : {total_mem_mb:.0f} MB")

    # 6. Create a tensor on GPU
    print("\nRunning GPU compute test...")
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")

    # 7. Matrix multiplication
    c = torch.matmul(a, b)

    # 8. Confirm result is on GPU
    assert c.device.type == "cuda", "Result tensor is not on CUDA device!"
    print(f"Result device  : {c.device}")
    print(f"Result shape   : {c.shape}")

    # 9 & 10. Summary
    free_mem_mb = torch.cuda.mem_get_info(0)[0] / 1024**2
    print(f"Free GPU memory: {free_mem_mb:.0f} MB")

    print("\n[SUCCESS] GPU is working correctly.")
    print("ForgeLM is ready for GPU-accelerated development.")
    print("=" * 50)


if __name__ == "__main__":
    main()
