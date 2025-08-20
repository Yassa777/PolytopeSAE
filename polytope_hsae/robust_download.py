"""
Robust model downloading with resume capability and connection monitoring.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import requests

logger = logging.getLogger(__name__)


class RobustModelDownloader:
    """Handles robust model downloading with resume capability."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_with_resume(self, model_name: str, max_retries: int = 5) -> bool:
        """Download model with resume capability and retries."""
        logger.info(f"🔄 Starting robust download of {model_name}")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"📥 Download attempt {attempt + 1}/{max_retries}")
                
                # Set resume download
                os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
                
                # Download tokenizer first (smaller, faster)
                logger.info("📝 Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    resume_download=True,
                    local_files_only=False
                )
                logger.info("✅ Tokenizer downloaded successfully")
                
                # Download model with resume
                logger.info("🤖 Downloading model...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    resume_download=True,
                    local_files_only=False,
                    torch_dtype=torch.float32,  # Avoid dtype issues
                    device_map=None  # Load to CPU first
                )
                logger.info("✅ Model downloaded successfully")
                
                # Verify model works
                logger.info("🧪 Testing model...")
                test_input = tokenizer("Hello world", return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**test_input)
                logger.info("✅ Model test successful")
                
                return True
                
            except Exception as e:
                logger.warning(f"❌ Download attempt {attempt + 1} failed: {e}")
                
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    logger.info("🔄 Connection issue detected, retrying...")
                elif "disk" in str(e).lower() or "space" in str(e).lower():
                    logger.error("💾 Disk space issue - cannot retry")
                    return False
                
                if attempt < max_retries - 1:
                    wait_time = min(60 * (2 ** attempt), 300)  # Exponential backoff, max 5min
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        logger.error(f"❌ Failed to download {model_name} after {max_retries} attempts")
        return False
    
    def check_model_availability(self, model_name: str) -> bool:
        """Check if model is already cached and working."""
        try:
            logger.info(f"🔍 Checking if {model_name} is already available...")
            
            # Try to load from cache
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=torch.float32,
                device_map=None
            )
            
            # Quick test
            test_input = tokenizer("Test", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**test_input)
            
            logger.info("✅ Model already available and working")
            return True
            
        except Exception as e:
            logger.info(f"📥 Model not cached or incomplete: {e}")
            return False
    
    def get_model_and_tokenizer(self, model_name: str, device: str = "cpu"):
        """Get model and tokenizer with robust downloading."""
        
        # Check if already available
        if not self.check_model_availability(model_name):
            # Download with resume capability
            if not self.download_with_resume(model_name):
                raise RuntimeError(f"Failed to download {model_name}")
        
        # Load model and tokenizer
        logger.info(f"📚 Loading {model_name} from cache...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        # Move to device
        if device != "cpu":
            model = model.to(device)
            
        logger.info(f"✅ {model_name} loaded successfully on {device}")
        return model, tokenizer


class ConnectionMonitor:
    """Monitors connection stability and provides warnings."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.last_check = time.time()
        self.connection_issues = 0
        
    def check_connection(self) -> bool:
        """Check internet connectivity."""
        try:
            response = requests.get("https://httpbin.org/status/200", timeout=10)
            if response.status_code == 200:
                self.connection_issues = 0
                return True
        except Exception:
            pass
        
        self.connection_issues += 1
        logger.warning(f"🌐 Connection issue detected ({self.connection_issues} consecutive)")
        return False
    
    def monitor_during_training(self, step: int) -> None:
        """Monitor connection during training and warn if issues detected."""
        current_time = time.time()
        
        if current_time - self.last_check > self.check_interval:
            if not self.check_connection():
                logger.warning(f"⚠️  Step {step}: Connection unstable - checkpoint recommended")
            self.last_check = current_time


def robust_model_setup(model_name: str, device: str = "cuda:0", cache_dir: Optional[str] = None):
    """Set up model with robust downloading and connection monitoring."""
    
    # Set up robust downloader
    downloader = RobustModelDownloader(cache_dir=cache_dir)
    
    # Get model and tokenizer
    model, tokenizer = downloader.get_model_and_tokenizer(model_name, device)
    
    # Set up connection monitor
    monitor = ConnectionMonitor()
    
    logger.info("🛡️ Robust model setup complete with connection monitoring")
    
    return model, tokenizer, monitor