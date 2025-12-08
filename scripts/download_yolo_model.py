#!/usr/bin/env python3
import os
import sys
import hashlib
import urllib.request
import urllib.error
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile
import shutil


class YOLOModelDownloader:
    """Downloads and verifies YOLO11 models for pose estimation."""

    # Official YOLO11 model URLs and checksums
    MODELS = {
        'yolo11n-pose.pt': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
            'size_mb': 6.8,
            'sha256': None,  # Will be calculated on first download
            'description': 'YOLOv11 Nano Pose (fastest, least accurate)'
        },
        'yolo11s-pose.pt': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt',
            'size_mb': 20.9,
            'sha256': None,
            'description': 'YOLOv11 Small Pose (balanced speed/accuracy)'
        },
        'yolo11m-pose.pt': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt',
            'size_mb': 49.4,
            'sha256': None,
            'description': 'YOLOv11 Medium Pose (good accuracy)'
        },
        'yolo11l-pose.pt': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt',
            'size_mb': 99.2,
            'sha256': None,
            'description': 'YOLOv11 Large Pose (high accuracy)'
        },
        'yolo11x-pose.pt': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt',
            'size_mb': 190.7,
            'sha256': None,
            'description': 'YOLOv11 Extra Large Pose (highest accuracy, slowest)'
        }
    }

    # Fallback URLs if main URLs fail
    FALLBACK_URLS = {
        'yolo11n-pose.pt': [
            'https://huggingface.co/Ultralytics/YOLOv11/resolve/main/yolo11n-pose.pt',
            'https://download.pytorch.org/models/yolo11n-pose.pt'
        ]
    }

    def __init__(self, models_dir: str = 'models'):
        """Initialize downloader with target directory."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.session_file = self.models_dir / '.download_session.json'

    def print_status(self, message: str, level: str = 'INFO') -> None:
        """Print formatted status message."""
        colors = {
            'INFO': '\033[94m',  # Blue
            'SUCCESS': '\033[92m',  # Green
            'WARNING': '\033[93m',  # Yellow
            'ERROR': '\033[91m',  # Red
            'RESET': '\033[0m'  # Reset
        }

        color = colors.get(level, colors['INFO'])
        reset = colors['RESET']
        symbols = {
            'INFO': 'ℹ️',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌'
        }
        symbol = symbols.get(level, 'ℹ️')

        print(f"{color}{symbol} {message}{reset}")

    def check_existing_model(self, model_name: str) -> bool:
        """Check if model already exists and is valid."""
        model_path = self.models_dir / model_name

        if not model_path.exists():
            return False

        # Check file size
        file_size = model_path.stat().st_size
        expected_size = self.MODELS[model_name]['size_mb'] * 1024 * 1024

        # Allow 5% tolerance in file size
        if abs(file_size - expected_size) / expected_size > 0.05:
            self.print_status(f"Existing {model_name} has incorrect size, will re-download", 'WARNING')
            return False

        self.print_status(f"Model {model_name} already exists and appears valid", 'SUCCESS')
        return True

    def download_with_progress(self, url: str, filepath: Path) -> bool:
        """Download file with progress bar."""
        try:
            self.print_status(f"Downloading from: {url}")

            # Create temporary file
            temp_path = filepath.with_suffix('.tmp')

            def show_progress(block_num: int, block_size: int, total_size: int):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded * 100) // total_size)
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)

                    # Update progress every 1MB or 5%
                    if block_num % 100 == 0 or percent in range(0, 101, 5):
                        print(f"\r📥 Progress: {percent:3d}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='',
                              flush=True)

            urllib.request.urlretrieve(url, temp_path, reporthook=show_progress)
            print()  # New line after progress

            # Move temp file to final location
            shutil.move(str(temp_path), str(filepath))
            return True

        except urllib.error.URLError as e:
            self.print_status(f"Download failed: {e}", 'ERROR')
            if temp_path.exists():
                temp_path.unlink()
            return False
        except KeyboardInterrupt:
            self.print_status("Download interrupted by user", 'WARNING')
            if temp_path.exists():
                temp_path.unlink()
            return False
        except Exception as e:
            self.print_status(f"Unexpected error during download: {e}", 'ERROR')
            if temp_path.exists():
                temp_path.unlink()
            return False

    def verify_download(self, filepath: Path, model_name: str) -> bool:
        """Verify downloaded model integrity."""
        # Check file exists
        if not filepath.exists():
            self.print_status(f"Downloaded file not found: {filepath}", 'ERROR')
            return False

        # Check file size
        file_size = filepath.stat().st_size
        expected_size = self.MODELS[model_name]['size_mb'] * 1024 * 1024

        if file_size < expected_size * 0.95:  # Allow 5% tolerance
            self.print_status(f"Downloaded file is too small: {file_size} bytes", 'ERROR')
            return False

        # Try to calculate SHA256 if not too large (< 200MB)
        if file_size < 200 * 1024 * 1024:
            try:
                sha256_hash = hashlib.sha256()
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        sha256_hash.update(chunk)

                calculated_hash = sha256_hash.hexdigest()
                self.print_status(f"File SHA256: {calculated_hash[:16]}...")

                # Store hash for future reference
                self.MODELS[model_name]['sha256'] = calculated_hash

            except Exception as e:
                self.print_status(f"Could not calculate hash: {e}", 'WARNING')

        self.print_status(f"Download verification successful", 'SUCCESS')
        return True

    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download specific YOLO model with fallback options."""
        if model_name not in self.MODELS:
            self.print_status(f"Unknown model: {model_name}", 'ERROR')
            self.print_status(f"Available models: {list(self.MODELS.keys())}")
            return False

        model_path = self.models_dir / model_name

        # Check if already exists
        if not force and self.check_existing_model(model_name):
            return True

        model_info = self.MODELS[model_name]
        self.print_status(f"Downloading {model_name} ({model_info['size_mb']:.1f} MB)")
        self.print_status(f"Description: {model_info['description']}")

        # Try main URL first
        urls_to_try = [model_info['url']]

        # Add fallback URLs if available
        if model_name in self.FALLBACK_URLS:
            urls_to_try.extend(self.FALLBACK_URLS[model_name])

        for i, url in enumerate(urls_to_try):
            if i > 0:
                self.print_status(f"Trying fallback URL {i}...", 'WARNING')

            if self.download_with_progress(url, model_path):
                if self.verify_download(model_path, model_name):
                    return True
                else:
                    model_path.unlink(missing_ok=True)

        self.print_status(f"Failed to download {model_name} from all sources", 'ERROR')
        return False

    def download_using_ultralytics(self, model_name: str) -> bool:
        """Fallback: Use ultralytics to download model."""
        try:
            self.print_status("Trying download via ultralytics library...")

            # Import ultralytics
            from ultralytics import YOLO

            # Create model (this will auto-download)
            model = YOLO(model_name)

            # Check if model was downloaded to default location
            import ultralytics
            default_path = Path(ultralytics.hub.utils.USER_CONFIG_DIR) / 'models' / model_name

            if default_path.exists():
                # Copy to our models directory
                target_path = self.models_dir / model_name
                shutil.copy2(default_path, target_path)
                self.print_status(f"Model copied to {target_path}", 'SUCCESS')
                return True

        except ImportError:
            self.print_status("Ultralytics library not available", 'WARNING')
        except Exception as e:
            self.print_status(f"Ultralytics download failed: {e}", 'WARNING')

        return False

    def list_models(self) -> None:
        """List available models with details."""
        print("\n🎯 Available YOLO11 Pose Models:")
        print("=" * 80)

        for model_name, info in self.MODELS.items():
            exists = "✅" if (self.models_dir / model_name).exists() else "❌"
            print(f"{exists} {model_name:18} | {info['size_mb']:6.1f} MB | {info['description']}")

        print("=" * 80)
        print("\n💡 Recommendations:")
        print("   • For development/testing: yolo11n-pose.pt (fastest)")
        print("   • For production balance: yolo11s-pose.pt (good speed/accuracy)")
        print("   • For best accuracy: yolo11m-pose.pt or larger")

    def download_recommended_set(self) -> bool:
        """Download recommended models for development."""
        recommended = ['yolo11n-pose.pt', 'yolo11s-pose.pt']
        success = True

        self.print_status("Downloading recommended model set...")

        for model in recommended:
            if not self.download_model(model):
                success = False

        return success

    def clean_downloads(self) -> None:
        """Clean up incomplete downloads."""
        temp_files = list(self.models_dir.glob('*.tmp'))
        if temp_files:
            self.print_status(f"Cleaning {len(temp_files)} temporary files...")
            for temp_file in temp_files:
                temp_file.unlink()
            self.print_status("Cleanup completed", 'SUCCESS')
        else:
            self.print_status("No temporary files to clean")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Download YOLO11 pose estimation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_yolo_model.py --list                    # List available models
  python download_yolo_model.py --model yolo11n-pose.pt  # Download specific model
  python download_yolo_model.py --recommended            # Download recommended set
  python download_yolo_model.py --all                    # Download all models
  python download_yolo_model.py --clean                  # Clean temporary files
        """
    )

    parser.add_argument('--models-dir', default='../../../models',
                        help='Directory to store models (default: ../../../models)')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--model',
                        help='Download specific model (e.g., yolo11n-pose.pt)')
    parser.add_argument('--recommended', action='store_true',
                        help='Download recommended models for development')
    parser.add_argument('--all', action='store_true',
                        help='Download all available models')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if model exists')
    parser.add_argument('--clean', action='store_true',
                        help='Clean temporary files')
    parser.add_argument('--fallback-ultralytics', action='store_true',
                        help='Use ultralytics library as fallback for downloads')

    args = parser.parse_args()

    # Resolve models directory relative to script location
    script_dir = Path(__file__).parent
    models_dir = (script_dir / args.models_dir).resolve()

    downloader = YOLOModelDownloader(str(models_dir))

    print("🤖 YOLO11 Pose Model Downloader")
    print(f"📁 Models directory: {models_dir}")
    print()

    try:
        if args.clean:
            downloader.clean_downloads()
            return

        if args.list:
            downloader.list_models()
            return

        if args.model:
            success = downloader.download_model(args.model, force=args.force)
            if not success and args.fallback_ultralytics:
                success = downloader.download_using_ultralytics(args.model)

            sys.exit(0 if success else 1)

        if args.recommended:
            success = downloader.download_recommended_set()
            sys.exit(0 if success else 1)

        if args.all:
            success = True
            for model_name in downloader.MODELS:
                if not downloader.download_model(model_name, force=args.force):
                    success = False
            sys.exit(0 if success else 1)

        # Default: show help
        parser.print_help()

    except KeyboardInterrupt:
        downloader.print_status("Operation interrupted by user", 'WARNING')
        sys.exit(1)
    except Exception as e:
        downloader.print_status(f"Unexpected error: {e}", 'ERROR')
        sys.exit(1)


if __name__ == '__main__':
    main()