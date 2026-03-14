"""
Backward-compatible wrapper for the unified dataset downloader.

This keeps old commands working:
  python download_dataset_hf_datasets.py

Equivalent to:
  python download_dataset.py --no-roboflow
"""

from download_dataset import build_dataset


def main() -> None:
    yaml_path = build_dataset(
        include_hf=True,
        include_roboflow=False,
        roboflow_api_key=None,
        roboflow_version=1,
        clean_temp=True,
    )
    print(f"[DONE] HF dataset prepared at: {yaml_path}")


if __name__ == "__main__":
    main()
