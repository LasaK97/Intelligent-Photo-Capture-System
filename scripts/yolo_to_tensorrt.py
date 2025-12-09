import os
import sys
import logging
import argparse
import threading
import subprocess
import re
from pathlib import Path
from typing import Optional, Tuple
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings, TensorRTEngineSettings
from mipcs.utils.logger import get_logger

logger = get_logger(__name__)

class YOLOModelConverter:
    MODEL_URLS = {
        "8": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
        "v8": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
        "9": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
        "v9": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
        "10": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
        "v10": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
        "11": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
        "v11": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
    }

    def __init__(self, config_path: Optional[str] = None):
        self.settings = get_settings()

        self.yolo_config = self.settings.vision.yolo
        self.model_path = Path(self.settings.vision.yolo.model_path)
        self.input_size = self.settings.vision.yolo.input_size

        engine_settings = self.yolo_config.tensorrt_engine_settings
        self.precision = engine_settings.precision
        self.batch_size = engine_settings.batch_size
        self.workspace_size = engine_settings.workspace_size

        model_source = self.yolo_config.model_source
        self.base_model = model_source.base_model
        self.download_url = model_source.download_url
        self.pt_model_path = Path(model_source.pt_model_path)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.pt_model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("YOLO TO TENSORRT ENGINE CONVERTER")
        logger.info("=" * 70)
        logger.info(f"Base model      : {self.base_model}")
        logger.info(f"PyTorch model   : {self.pt_model_path}")
        logger.info(f"TensorRT engine : {self.model_path}")
        logger.info(f"Input size      : {self.input_size}")
        logger.info(f"Precision       : {self.precision}")
        logger.info(f"Batch size      : {self.batch_size}")
        logger.info(f"Workspace       : {self.workspace_size}GB")
        logger.info("=" * 70)

    def _parse_model_name(self) -> Tuple[str, str, str, str]:
        pattern = r'^yolo(v)?(\d{1,2})(n|s|m|l|x)-(pose|seg|detect|obb|cls)$'
        match = re.match(pattern, self.base_model.lower())

        if not match:
            raise ValueError(
                f"Invalid YOLO model name. {self.base_model} \n."
                f"Expected format: yolo[v]<version><size>-<task>\n"
                f"Examples: yolo11n-pose, yolov8m-pose, yolo10s-seg"
            )
        has_v, version, size, task = match.groups()
        return version, size, task, self.base_model.lower()

    def _get_download_url(self) -> str:
        if self.download_url:
            logger.info(f"Using custom URL: {self.download_url}")
            return self.download_url

        version, size, task, full_name = self._parse_model_name()

        if version in self.MODEL_URLS:
            base_url = self.MODEL_URLS[version]
        elif f"v{version}" in self.MODEL_URLS:
            base_url = self.MODEL_URLS[f"v{version}"]
        else:
            logger.warning(f"Unknown YOLO version: {version}. Using v11 URL")
            base_url = self.MODEL_URLS["11"]

        model_filename = f"{full_name}.pt"
        download_url = f"{base_url}/{model_filename}"

        return download_url

    def check_dependencies(self) -> bool:
        required_packages = {
            "ultralytics": "ultralytics",
            "torch": "torch",
            "tensorrt": "tensorrt",
            "tqdm": "tqdm",
        }

        missing = []
        for package, import_name in required_packages.items():
            try:
                __import__(import_name)
                logger.info(f"✓ {package}")
            except ImportError:
                missing.append(package)
                logger.error(f"✗ {package}")

        if missing:
            logger.error(f"Missing packages: {', '.join(missing)}")
            logger.error("Install: pip install " + " ".join(missing))
            return False

        return True

    def check_cuda(self) -> bool:
        try:
            if not torch.cuda.is_available():
                logger.error("CUDA not available")
                return False

            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda

            logger.info(f"✓ CUDA {cuda_version}")
            logger.info(f"✓ {device_name}")
            logger.info(f"✓ {device_count} GPU(s) available")

            return True
        except Exception as e:
            logger.error(f"CUDA check failed: {e}")
            return False

    def download_model(self, force: bool = False) -> bool:
        # Check existing file and verify integrity
        if self.pt_model_path.exists() and not force:
            if self._verify_pt_model():
                logger.info(f"✓ Model exists: {self.pt_model_path}")
                return True
            else:
                logger.warning(f"⚠ Corrupted model, re-downloading...")
                self.pt_model_path.unlink()

        logger.info(f"Downloading {self.base_model}...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                import warnings
                warnings.filterwarnings('ignore')

                os.environ['YOLO_VERBOSE'] = 'False'

                from ultralytics import YOLO

                # Clean up corrupted file
                if self.pt_model_path.exists():
                    self.pt_model_path.unlink()

                # Ensure directory exists
                self.pt_model_path.parent.mkdir(parents=True, exist_ok=True)

                # Download model
                model = YOLO(f"{self.base_model}.pt")

                possible_paths = [
                    Path(f"{self.base_model}.pt").absolute(),  # Current directory
                    Path.home() / ".cache" / "ultralytics" / f"{self.base_model}.pt",
                ]

                source_path = None
                for path in possible_paths:
                    if path.exists():
                        source_path = path
                        logger.info(f"Found downloaded file: {source_path}")
                        break

                if source_path:
                    import shutil
                    logger.info(f"Moving to: {self.pt_model_path}")
                    shutil.move(str(source_path), str(self.pt_model_path))
                else:
                    logger.info(f"Saving to: {self.pt_model_path}")
                    model.save(str(self.pt_model_path))

                # Verify file exists and is valid
                if not self.pt_model_path.exists():
                    raise FileNotFoundError("Model file not created")

                size_mb = self.pt_model_path.stat().st_size / (1024 * 1024)
                logger.info(f"File size: {size_mb:.2f} MB")

                # Verify integrity
                if size_mb < 1.0:
                    raise ValueError(f"File too small: {size_mb:.2f} MB")

                if not self._verify_pt_model():
                    raise ValueError("Model integrity check failed")

                logger.info(f"✓ Downloaded: {self.pt_model_path} ({size_mb:.2f} MB)")
                return True

            except Exception as e:
                # Clean up any leftover files
                if self.pt_model_path.exists():
                    self.pt_model_path.unlink()

                current_file = Path(f"{self.base_model}.pt")
                if current_file.exists():
                    current_file.unlink()

                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    import time
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"Download error: {e}")
                    return False

        return False

    def _verify_pt_model(self) -> bool:
        """Verify PyTorch model integrity"""
        if not self.pt_model_path.exists():
            return False

        try:
            import torch
            size_bytes = self.pt_model_path.stat().st_size
            if size_bytes < 1_000_000:
                logger.warning(f"File too small: {size_bytes} bytes")
                return False

            logger.info("Verifying model integrity...")

            checkpoint = torch.load(
                self.pt_model_path,
                map_location='cpu',
                weights_only=False
            )

            if not isinstance(checkpoint, dict):
                logger.warning("Invalid checkpoint format")
                return False

            if 'model' not in checkpoint:
                logger.warning("Missing 'model' key in checkpoint")
                return False

            logger.info("✓ Model integrity verified")
            return True
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return False

    def convert_to_tensorrt(self, force: bool = False) -> bool:
        if self.model_path.exists() and not force:
            logger.info(f"✓ Engine exists: {self.model_path}")
            logger.info("Use --force-convert to rebuild")
            return True

        if not self.pt_model_path.exists():
            logger.error(f"PyTorch model not found: {self.pt_model_path}")
            return False

        import time
        start_time = time.time()

        try:
            import warnings
            warnings.filterwarnings('ignore')
            os.environ['YOLO_VERBOSE'] = 'False'

            from ultralytics import YOLO

            logger.info("-" * 70)
            logger.info("Converting to TensorRT with Ultralytics export")

            model = YOLO(str(self.pt_model_path))

            export_args = {
                "format": "engine",
                "imgsz": self.input_size,
                "batch": self.batch_size,
                "workspace": self.workspace_size,
                "verbose": True,
            }

            if self.precision == "FP16":
                export_args["half"] = True
                logger.info("✓ FP16 precision enabled")
            elif self.precision == "INT8":
                export_args["int8"] = True
                logger.warning("INT8 precision requires calibration data")

            logger.info("Building TensorRT engine...")
            export_path = model.export(**export_args)

            export_path = Path(export_path)
            if export_path != self.model_path:
                logger.info(f"Moving engine to: {self.model_path}")
                import shutil
                shutil.move(str(export_path), str(self.model_path))

            engine_size_mb = self.model_path.stat().st_size / (1024 * 1024)
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            logger.info(f"✓ Engine saved ({engine_size_mb:.2f} MB)")
            logger.info(f"✓ Time: {minutes}m {seconds}s")

            return True

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def verify_engine(self) -> bool:
        if not self.model_path.exists():
            logger.error(f"Engine not found: {self.model_path}")
            return False

        logger.info("Verifying engine...")

        try:
            import warnings
            warnings.filterwarnings('ignore')
            os.environ['YOLO_VERBOSE'] = 'False'

            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
            logger.info(f"✓ Engine verified ({getattr(model, 'task', 'unknown')})")

            return True

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def run(
            self,
            force_download: bool = False,
            force_convert: bool = False,
            download_only: bool = False,
            skip_verify: bool = False
    ) -> bool:

        logger.info("")
        logger.info("[PHASE 1] Dependency Check")
        if not self.check_dependencies():
            return False

        logger.info("")
        logger.info("[PHASE 2] CUDA Check")
        if not self.check_cuda():
            return False

        logger.info("")
        logger.info("[PHASE 3] Model Download")
        if not self.download_model(force=force_download):
            return False

        if download_only:
            logger.info("")
            logger.info("✓ Download complete")
            return True

        logger.info("")
        logger.info("[PHASE 4] TensorRT Conversion")
        if not self.convert_to_tensorrt(force=force_convert):
            return False

        if not skip_verify:
            logger.info("")
            logger.info("[PHASE 5] Verification")
            if not self.verify_engine():
                logger.warning("Verification failed")

        logger.info("")
        logger.info("=" * 70)
        logger.info("CONVERSION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Engine : {self.model_path}")
        logger.info(f"Usage  : from ultralytics import YOLO")
        logger.info(f"         model = YOLO('{self.model_path}')")
        logger.info("=" * 70)

        return True


def main():
    logger = get_logger()

    parser = argparse.ArgumentParser(
        description="Convert YOLO models to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--force-download", action="store_true", help="Force re-download")
    parser.add_argument("--force-convert", action="store_true", help="Force re-conversion")
    parser.add_argument("--force-all", action="store_true", help="Force both")
    parser.add_argument("--download-only", action="store_true", help="Only download")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        converter = YOLOModelConverter(config_path=args.config)

        success = converter.run(
            force_download=args.force_download or args.force_all,
            force_convert=args.force_convert or args.force_all,
            download_only=args.download_only,
            skip_verify=args.skip_verify
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()