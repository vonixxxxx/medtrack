"""
Automatic dataset downloader for ePillID benchmark
Downloads from GitHub releases and handles extraction
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import json
from tqdm import tqdm
import shutil


class DatasetDownloader:
    """
    Automatically downloads and extracts ePillID dataset from GitHub releases.
    """
    
    GITHUB_REPO = "usuyama/ePillID-benchmark"
    RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
    
    def __init__(self, download_dir: str = "./data/epillid"):
        """
        Initialize downloader.
        
        Args:
            download_dir: Directory to download and extract dataset
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def get_latest_release(self) -> Optional[dict]:
        """Get latest release information from GitHub"""
        try:
            response = requests.get(self.RELEASE_API, timeout=10)
            response.raise_for_status()
            releases = response.json()
            
            if not releases:
                print("No releases found")
                return None
            
            # Get latest release
            latest = releases[0]
            return latest
        except Exception as e:
            print(f"Error fetching releases: {e}")
            return None
    
    def find_dataset_asset(self, release: dict) -> Optional[dict]:
        """Find dataset asset in release (ZIP or TAR)"""
        assets = release.get('assets', [])
        
        # Look for dataset files
        dataset_keywords = ['data', 'dataset', 'epillid', 'v1', 'v1.0']
        
        for asset in assets:
            name = asset['name'].lower()
            # Check if it's an archive
            if any(name.endswith(ext) for ext in ['.zip', '.tar', '.tar.gz', '.tgz']):
                # Check if it looks like the dataset
                if any(keyword in name for keyword in dataset_keywords):
                    return asset
        
        # Fallback: return first archive
        for asset in assets:
            name = asset['name'].lower()
            if any(name.endswith(ext) for ext in ['.zip', '.tar', '.tar.gz', '.tgz']):
                return asset
        
        return None
    
    def download_file(self, url: str, filepath: Path, expected_size: Optional[int] = None) -> bool:
        """
        Download a file with progress bar.
        
        Args:
            url: Download URL
            filepath: Local file path
            expected_size: Expected file size in bytes
        """
        try:
            print(f"Downloading from: {url}")
            print(f"Destination: {filepath}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = expected_size or int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"✓ Download complete: {filepath}")
            return True
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Extract archive (ZIP or TAR).
        
        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to
        """
        try:
            print(f"Extracting {archive_path.name}...")
            
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # Get total files for progress
                    members = zip_ref.namelist()
                    total = len(members)
                    
                    with tqdm(total=total, desc="Extracting") as pbar:
                        for member in members:
                            zip_ref.extract(member, extract_to)
                            pbar.update(1)
            
            elif archive_path.suffix in ['.tar', '.gz'] or '.tar.gz' in archive_path.name:
                mode = 'r:gz' if '.gz' in archive_path.name else 'r'
                with tarfile.open(archive_path, mode) as tar_ref:
                    tar_ref.extractall(extract_to)
            
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            print(f"✓ Extraction complete: {extract_to}")
            return True
            
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    def validate_dataset(self, dataset_path: Path) -> Tuple[bool, dict]:
        """
        Validate dataset structure and integrity.
        
        Args:
            dataset_path: Path to extracted dataset
        
        Returns:
            Tuple of (is_valid, stats_dict)
        """
        stats = {
            'total_images': 0,
            'directories': 0,
            'csv_files': 0,
            'structure': 'unknown'
        }
        
        if not dataset_path.exists():
            return False, stats
        
        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_count = 0
        
        for root, dirs, files in os.walk(dataset_path):
            stats['directories'] += len(dirs)
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_count += 1
                elif file.lower().endswith('.csv'):
                    stats['csv_files'] += 1
        
        stats['total_images'] = image_count
        
        # Determine structure
        if stats['csv_files'] > 0:
            stats['structure'] = 'csv_based'
        elif image_count > 0:
            stats['structure'] = 'folder_based'
        
        # Validation: should have images
        is_valid = image_count > 0
        
        return is_valid, stats
    
    def find_dataset_root(self, extract_dir: Path) -> Optional[Path]:
        """
        Find the actual dataset root directory.
        Sometimes archives extract to a subdirectory.
        """
        # Look for common dataset indicators
        indicators = ['data', 'images', 'classification_data', 'epillid']
        
        # Check if extract_dir itself is the dataset
        if any((extract_dir / ind).exists() for ind in indicators):
            return extract_dir
        
        # Check subdirectories
        for item in extract_dir.iterdir():
            if item.is_dir():
                if any((item / ind).exists() for ind in indicators):
                    return item
                # Check if it has images directly
                image_count = sum(
                    1 for f in item.rglob('*')
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                )
                if image_count > 10:  # Likely the dataset
                    return item
        
        # Fallback: return extract_dir
        return extract_dir
    
    def download_and_extract(self, force: bool = False) -> Optional[Path]:
        """
        Download and extract dataset automatically.
        
        Args:
            force: Force re-download even if exists
        
        Returns:
            Path to extracted dataset, or None if failed
        """
        # Check if dataset already exists
        dataset_path = self.download_dir / "dataset"
        if dataset_path.exists() and not force:
            print(f"Dataset already exists at: {dataset_path}")
            is_valid, stats = self.validate_dataset(dataset_path)
            if is_valid:
                print(f"✓ Dataset validated: {stats['total_images']} images found")
                return dataset_path
            else:
                print("⚠ Existing dataset appears invalid, re-downloading...")
        
        # Get latest release
        print("Fetching latest release information...")
        release = self.get_latest_release()
        if not release:
            print("✗ Could not fetch release information")
            return None
        
        print(f"Latest release: {release['tag_name']} - {release['name']}")
        
        # Find dataset asset
        asset = self.find_dataset_asset(release)
        if not asset:
            print("✗ No dataset asset found in release")
            print("Available assets:")
            for a in release.get('assets', []):
                print(f"  - {a['name']}")
            return None
        
        print(f"Found dataset asset: {asset['name']} ({asset['size'] / (1024*1024):.1f} MB)")
        
        # Download
        archive_path = self.download_dir / asset['name']
        
        if archive_path.exists() and not force:
            print(f"Archive already exists: {archive_path}")
        else:
            if not self.download_file(asset['browser_download_url'], archive_path, asset['size']):
                return None
        
        # Extract
        extract_dir = self.download_dir / "extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        if not self.extract_archive(archive_path, extract_dir):
            return None
        
        # Find actual dataset root
        dataset_root = self.find_dataset_root(extract_dir)
        if not dataset_root:
            print("✗ Could not find dataset root")
            return None
        
        # Move to final location
        final_path = self.download_dir / "dataset"
        if final_path.exists():
            shutil.rmtree(final_path)
        
        if dataset_root != extract_dir:
            shutil.move(str(dataset_root), str(final_path))
        else:
            # Move entire extract_dir
            extract_dir.rename(final_path)
        
        # Validate
        is_valid, stats = self.validate_dataset(final_path)
        if not is_valid:
            print("✗ Dataset validation failed")
            return None
        
        print(f"✓ Dataset ready: {stats['total_images']} images, {stats['csv_files']} CSV files")
        print(f"  Structure: {stats['structure']}")
        print(f"  Location: {final_path}")
        
        return final_path


def download_dataset(download_dir: str = "./data/epillid", force: bool = False) -> Optional[Path]:
    """
    Convenience function to download dataset.
    
    Args:
        download_dir: Directory to download dataset
        force: Force re-download
    
    Returns:
        Path to dataset, or None if failed
    """
    downloader = DatasetDownloader(download_dir)
    return downloader.download_and_extract(force=force)







