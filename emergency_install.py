#!/usr/bin/env python3
"""
Emergency Installation Script for RunPod
Handles pip timeouts by installing packages individually with retries
"""

import subprocess
import sys
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Essential packages in order of importance
ESSENTIAL_PACKAGES = [
    "torch>=2.1.0",
    "numpy>=1.24.0", 
    "transformers>=4.41.0",
    "accelerate>=0.28.0",
    "sentencepiece>=0.1.99",
    "safetensors>=0.4.3",
    "huggingface_hub>=0.23.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "h5py>=3.8.0",
    "pyyaml>=6.0",
    "wandb>=0.15.0",
    "tqdm>=4.65.0",
    "nltk>=3.8.0",
    "datasets>=2.12.0",
    "tokenizers>=0.13.0"
]

def run_command(cmd, timeout=300, retries=3):
    """Run a command with timeout and retries."""
    for attempt in range(retries):
        try:
            logger.info(f"Running: {' '.join(cmd)} (attempt {attempt + 1}/{retries})")
            result = subprocess.run(
                cmd, 
                timeout=timeout, 
                capture_output=True, 
                text=True,
                check=True
            )
            logger.info("✅ Command succeeded")
            return True, result.stdout
        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ Command timed out after {timeout}s")
        except subprocess.CalledProcessError as e:
            logger.warning(f"❌ Command failed: {e.stderr}")
        except Exception as e:
            logger.warning(f"❌ Unexpected error: {e}")
        
        if attempt < retries - 1:
            wait_time = (attempt + 1) * 30
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    return False, None

def install_package(package, strategies=None):
    """Try multiple installation strategies for a package."""
    if strategies is None:
        strategies = [
            # Strategy 1: Standard pip with increased timeout
            ["pip", "install", "--timeout", "300", "--retries", "5", package],
            
            # Strategy 2: No cache
            ["pip", "install", "--timeout", "300", "--retries", "5", "--no-cache-dir", package],
            
            # Strategy 3: Different index
            ["pip", "install", "--timeout", "300", "--retries", "5", 
             "-i", "https://pypi.python.org/simple/", package],
            
            # Strategy 4: Force reinstall
            ["pip", "install", "--timeout", "300", "--retries", "5", 
             "--force-reinstall", "--no-deps", package],
        ]
    
    logger.info(f"🔄 Installing {package}...")
    
    for i, strategy in enumerate(strategies):
        logger.info(f"Trying strategy {i + 1}/{len(strategies)}")
        success, output = run_command(strategy)
        if success:
            logger.info(f"✅ Successfully installed {package}")
            return True
    
    logger.error(f"❌ Failed to install {package} with all strategies")
    return False

def check_installation():
    """Test that key imports work."""
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import transformers; print(f'Transformers: {transformers.__version__}')",
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import pandas; print(f'Pandas: {pandas.__version__}')",
        "import h5py; print(f'HDF5: {h5py.__version__}')",
        "import wandb; print(f'W&B: {wandb.__version__}')",
    ]
    
    logger.info("🧪 Testing installations...")
    for test in test_imports:
        try:
            result = subprocess.run([sys.executable, "-c", test], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Import test failed: {e.stderr}")
            return False
    
    return True

def main():
    """Main installation process."""
    logger.info("🚀 Starting Emergency Installation Process")
    logger.info("=" * 50)
    
    # Upgrade pip first
    logger.info("📦 Upgrading pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install packages one by one
    failed_packages = []
    for package in ESSENTIAL_PACKAGES:
        if not install_package(package):
            failed_packages.append(package)
            
            # For critical packages, try alternative approaches
            if any(critical in package for critical in ["torch", "transformers", "numpy"]):
                logger.warning(f"Critical package {package} failed, trying alternatives...")
                
                # Try installing without version constraints
                base_package = package.split(">=")[0].split("==")[0]
                if install_package(base_package):
                    logger.info(f"✅ Installed {base_package} without version constraint")
                    failed_packages.remove(package)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 INSTALLATION SUMMARY")
    logger.info("=" * 50)
    
    successful = len(ESSENTIAL_PACKAGES) - len(failed_packages)
    logger.info(f"✅ Successfully installed: {successful}/{len(ESSENTIAL_PACKAGES)} packages")
    
    if failed_packages:
        logger.warning("❌ Failed packages:")
        for pkg in failed_packages:
            logger.warning(f"  - {pkg}")
    
    # Test installations
    if check_installation():
        logger.info("🎉 All critical imports working!")
        
        # Install PolytopeSAE package
        logger.info("📦 Installing PolytopeSAE package...")
        if install_package("-e ."):
            logger.info("🎉 PolytopeSAE installation complete!")
            return True
        else:
            logger.error("❌ PolytopeSAE package installation failed")
            return False
    else:
        logger.error("❌ Critical imports failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)