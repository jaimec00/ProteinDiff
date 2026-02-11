
import argparse
import tarfile
from enum import StrEnum, auto
from pathlib import Path

import requests

MPNN_URL = "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz"


def download_mpnn(data_dir: Path) -> None:
    """download and extract the ProteinMPNN training dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "pdb_2021aug02.tar.gz"

    # stream download the tarball with progress
    print(f"downloading ProteinMPNN dataset to {tar_path}...")
    response = requests.get(MPNN_URL, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / total:.0%}", end="", flush=True)

    print()

    # extract and clean up
    print(f"extracting to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")

    tar_path.unlink()
    print("done.")


class Dataset(StrEnum):
    MPNN = auto()


DOWNLOAD_FNS = {
    Dataset.MPNN: download_mpnn,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download proteus training datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[d.value for d in Dataset],
        required=True,
        help="dataset to download",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data"),
        help="directory to download data to",
    )
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    DOWNLOAD_FNS[dataset](args.data_dir)
