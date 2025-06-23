#!/usr/bin/env python3
"""
Script to check PyTorch GPU availability and CUDA configuration
"""

import subprocess
import sys

def check_pytorch_gpu():
    """Check PyTorch GPU availability and provide detailed information"""
    
    print("=" * 60)
    print("PyTorch GPU Availability Check")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            # CUDA version
            print(f"✓ CUDA version: {torch.version.cuda}")
            
            # cuDNN version
            if torch.backends.cudnn.enabled:
                print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
                print(f"✓ cuDNN enabled: {torch.backends.cudnn.enabled}")
            else:
                print("✗ cuDNN not available")
            
            # Number of GPUs
            gpu_count = torch.cuda.device_count()
            print(f"✓ Number of GPUs: {gpu_count}")
            
            # GPU details
            print("\n--- GPU Details ---")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                gpu_capability = torch.cuda.get_device_capability(i)
                
                print(f"  GPU {i}: {gpu_name}")
                print(f"    Memory: {gpu_memory_gb:.1f} GB")
                print(f"    Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
                
                # Current memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    print(f"    Memory Allocated: {allocated:.2f} GB")
                    print(f"    Memory Cached: {cached:.2f} GB")
            
            # Current device
            current_device = torch.cuda.current_device()
            print(f"\n✓ Current CUDA device: {current_device}")
            
            # Test GPU computation
            print("\n--- GPU Computation Test ---")
            try:
                device = torch.device('cuda')
                
                # Create tensors on GPU
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                
                # Perform matrix multiplication
                c = torch.matmul(a, b)
                
                # Move result back to CPU to verify
                result_cpu = c.cpu()
                
                print("✓ GPU tensor creation: Success")
                print("✓ GPU matrix multiplication: Success")
                print(f"  Result shape: {result_cpu.shape}")
                print(f"  Result sample: {result_cpu[0, 0].item():.4f}")
                
                # Test if model can be moved to GPU
                from torch import nn
                model = nn.Linear(10, 1)
                model = model.to(device)
                print("✓ Model transfer to GPU: Success")
                
                # Test forward pass
                test_input = torch.randn(5, 10, device=device)
                output = model(test_input)
                print(f"✓ GPU forward pass: Success (output shape: {output.shape})")
                
            except Exception as e:
                print(f"✗ GPU computation test failed: {e}")
        
        else:
            print("✗ CUDA is not available")
            print("\nReasons CUDA might not be available:")
            print("1. No NVIDIA GPU detected")
            print("2. NVIDIA drivers not installed")
            print("3. CUDA toolkit not installed")
            print("4. PyTorch not compiled with CUDA support")
            print("5. GPU compute capability too old")
    
    except ImportError:
        print("✗ PyTorch is not installed")
        print("To install PyTorch with CUDA support, visit:")
        print("  https://pytorch.org/get-started/locally/")
        return False
    
    except Exception as e:
        print(f"✗ Error checking PyTorch GPU: {e}")
        return False
    
    # Check NVIDIA System Management Interface
    print("\n--- NVIDIA System Info ---")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ nvidia-smi output:")
            lines = result.stdout.split('\n')
            for line in lines[:15]:  # Show first 15 lines
                if line.strip():
                    print(f"  {line}")
            if len(lines) > 15:
                print("  ... (output truncated)")
        else:
            print("✗ nvidia-smi command failed")
    except subprocess.TimeoutExpired:
        print("⚠ nvidia-smi command timed out")
    except FileNotFoundError:
        print("✗ nvidia-smi not found - NVIDIA drivers may not be installed")
    except Exception as e:
        print(f"⚠ Could not run nvidia-smi: {e}")
    
    # Summary
    print("\n--- Summary ---")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✓ PyTorch can use {gpu_count} GPU(s) for training")
            print(f"  Primary GPU: {current_gpu}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print("\nYou can use GPU for training by:")
            print("  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
            print("  model = model.to(device)")
            print("  data = data.to(device)")
        else:
            print("✗ PyTorch will use CPU for training")
            print("\nTo enable GPU training:")
            print("1. Install NVIDIA GPU drivers")
            print("2. Install CUDA toolkit")
            print("3. Ensure PyTorch was installed with CUDA support")
    except:
        pass
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    check_pytorch_gpu() 