#!/usr/bin/env python3
"""
Robust model download using aria2c for multi-connection downloads.
Handles HuggingFace model downloads with resume capability and connection splitting.
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict
import requests
from huggingface_hub import HfApi

def get_model_files(repo_id: str) -> List[Dict]:
    """Get list of files in HuggingFace repository."""
    print(f"🔍 Getting file list for {repo_id}...")
    
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id)
        files = []
        
        for sibling in repo_info.siblings:
            if not sibling.rfilename.startswith('.'):  # Skip hidden files
                file_size = getattr(sibling, 'size', None)
                if file_size is None:
                    file_size = 0  # Default to 0 if size is None
                
                files.append({
                    'filename': sibling.rfilename,
                    'size': file_size,
                    'url': f"https://huggingface.co/{repo_id}/resolve/main/{sibling.rfilename}"
                })
        
        # Sort by size (download small files first), handle None values
        files.sort(key=lambda x: x['size'] if x['size'] is not None else 0)
        return files
        
    except Exception as e:
        print(f"❌ Error getting file list: {e}")
        print("🔄 Using fallback file list...")
        
        # Fallback: common HuggingFace model files
        common_files = [
            'config.json',
            'tokenizer.json', 
            'tokenizer_config.json',
            'vocab.json',
            'merges.txt',
            'pytorch_model.bin'
        ]
        
        files = []
        for filename in common_files:
            files.append({
                'filename': filename,
                'size': 0,  # Unknown size
                'url': f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            })
        
        print(f"📋 Using {len(files)} common files")
        return files

def download_with_aria2c(url: str, output_path: str, connections: int = 8, max_retries: int = 5) -> bool:
    """Download file using aria2c with multiple connections."""
    
    cmd = [
        'aria2c',
        '--continue=true',                    # Resume downloads
        f'--max-connection-per-server={connections}',  # Multiple connections
        f'--split={connections}',             # Split into segments
        '--min-split-size=1M',               # Minimum segment size
        '--max-tries=10',                    # Retry attempts
        '--retry-wait=30',                   # Wait between retries
        '--timeout=60',                      # Connection timeout
        '--max-download-limit=0',            # No speed limit
        '--check-certificate=false',         # Skip cert check if needed
        '--summary-interval=10',             # Progress updates
        f'--dir={os.path.dirname(output_path)}',  # Output directory
        f'--out={os.path.basename(output_path)}', # Output filename
        '--user-agent=Mozilla/5.0 (compatible; aria2c)',  # User agent
        url
    ]
    
    print(f"📥 Downloading {os.path.basename(output_path)} with {connections} connections...")
    print(f"🔗 Command: {' '.join(cmd)}")
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                print(f"✅ Successfully downloaded {os.path.basename(output_path)}")
                return True
            else:
                print(f"❌ aria2c failed (attempt {attempt + 1}/{max_retries})")
                print(f"stderr: {result.stderr}")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # Exponential backoff
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ Download timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print("🔄 Retrying with resume...")
                time.sleep(60)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
    
    return False

def download_huggingface_model(repo_id: str, cache_dir: str = "/workspace/hf_cache"):
    """Download complete HuggingFace model using aria2c."""
    
    print(f"🚀 Starting robust download of {repo_id}")
    print(f"📁 Cache directory: {cache_dir}")
    
    # Create cache directory
    model_cache_dir = Path(cache_dir) / repo_id.replace('/', '--')
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get file list
    files = get_model_files(repo_id)
    if not files:
        print("❌ Could not get file list")
        return False
    
    print(f"📋 Found {len(files)} files to download")
    
    # Download each file
    success_count = 0
    for i, file_info in enumerate(files, 1):
        filename = file_info['filename']
        url = file_info['url']
        size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] > 0 else 0
        
        output_path = model_cache_dir / filename
        
        print(f"\n📦 [{i}/{len(files)}] {filename} ({size_mb:.1f} MB)")
        
        # Skip if already exists and complete
        if output_path.exists() and output_path.stat().st_size == file_info['size']:
            print(f"✅ Already downloaded: {filename}")
            success_count += 1
            continue
        
        # Choose connection count based on file size
        connections = min(16, max(1, int(size_mb / 10)))  # 1 connection per 10MB, max 16
        
        success = download_with_aria2c(url, str(output_path), connections)
        if success:
            success_count += 1
        else:
            print(f"❌ Failed to download {filename}")
            # Continue with other files
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {success_count}/{len(files)}")
    print(f"❌ Failed: {len(files) - success_count}/{len(files)}")
    
    if success_count >= len(files) * 0.8:  # 80% success rate
        print(f"🎉 Model download completed successfully!")
        print(f"📁 Model saved to: {model_cache_dir}")
        return True
    else:
        print(f"❌ Too many failures, model incomplete")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python download_model_aria2c.py <model_name> [cache_dir]")
        print("Example: python download_model_aria2c.py distilgpt2")
        sys.exit(1)
    
    model_name = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/hf_cache"
    
    success = download_huggingface_model(model_name, cache_dir)
    sys.exit(0 if success else 1)