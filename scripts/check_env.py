#!/usr/bin/env python3
"""Environment and Hardware Verification Script for SkinRisk Fusion."""

import sys
import platform

def run_environment_check():
    print("=" * 60)
    print("SkinRisk Fusion — Environment & Hardware Verification")
    print("=" * 60)

    # 1. Python & System details
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"OS & Machine     : {platform.system()} {platform.release()} ({platform.machine()})")
    print("-" * 60)

    # 2. Critical library imports & versions
    libraries = [
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("lightgbm", "LightGBM"),
        ("h5py", "H5Py"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("pytest", "Pytest"),
        ("yaml", "PyYAML"),
    ]

    all_passed = True
    print("Library Import Status:")
    for mod_name, display_name in libraries:
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK]   {display_name:<15} version: {version}")
        except ImportError as e:
            print(f"  [FAIL] {display_name:<15} Error: {e}")
            all_passed = False

    print("-" * 60)

    # 3. PyTorch Hardware & Acceleration Detection
    print("Hardware Acceleration Check:")
    try:
        import torch

        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()

        print(f"  torch.backends.mps.is_built()     : {mps_built}")
        print(f"  torch.backends.mps.is_available() : {mps_available}")

        # Check preferred device
        if mps_available:
            device = torch.device("mps")
            # Quick test: allocate tiny 2x2 tensor to verify MPS kernel execution
            x = torch.ones((2, 2), device=device)
            y = x * 2.0
            print(f"  Primary Device Selected           : MPS (Apple Silicon GPU)")
            print(f"  MPS Tensor Execution Test         : PASSED (sum={y.sum().item()})")
        else:
            device = torch.device("cpu")
            print(f"  Primary Device Selected           : CPU (Fallback)")

        # Verify CPU fallback is always operational
        cpu_x = torch.ones((2, 2), device=torch.device("cpu"))
        print(f"  CPU Device Fallback               : Operational (sum={cpu_x.sum().item()})")

    except Exception as e:
        print(f"  [FAIL] PyTorch Acceleration check failed: {e}")
        all_passed = False

    print("-" * 60)

    # 4. HDF5 sanity check
    try:
        import h5py
        print(f"HDF5 File Interface: Operational (API version {h5py.version.hdf5_version})")
    except Exception as e:
        print(f"  [FAIL] HDF5 check failed: {e}")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL ENVIRONMENT & HARDWARE CHECKS PASSED SUCCESSFULLY.")
        return 0
    else:
        print("SOME CHECKS FAILED. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_environment_check())
