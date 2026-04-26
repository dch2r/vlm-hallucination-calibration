"""
MS COCO 2014 validation loader for hallucination evaluation.

Provides a minimal, no-dependency loader that:
  1. Downloads the COCO 2014 instance annotations (~250MB) on first use.
  2. Streams a configurable number of (image, ground_truth_objects) pairs
     by lazily fetching individual images from the public COCO image URLs.

We download images one at a time from images.cocodataset.org rather than
the full 6GB val2014.zip to keep storage and bandwidth low. This is
appropriate for a research evaluation set of 50-1000 images.

Ground-truth objects are mapped from COCO's 80 category IDs to their
plain English names (e.g. "person", "bicycle", "dining table").
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set
from urllib.request import urlretrieve

from PIL import Image


COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMG_URL_TEMPLATE = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

DEFAULT_CACHE_DIR = Path(os.environ.get("COCO_CACHE", str(Path.home() / ".cache" / "coco")))


@dataclass
class COCOSample:
    """One COCO val image + its ground-truth object set."""
    image_id: int
    image_path: Path
    image: Image.Image
    gt_objects: Set[str]            # Plain English names, e.g. {"person", "bicycle"}
    gt_object_ids: Set[int]         # Raw COCO category IDs


class COCOLoader:
    """
    Lazy loader for COCO 2014 val images and ground-truth objects.

    Example:
        loader = COCOLoader()
        for sample in loader.iter_samples(n=10):
            print(sample.image_id, sample.gt_objects)
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.cache_dir / "val2014"
        self.images_dir.mkdir(exist_ok=True)
        self.ann_path = self.cache_dir / "annotations" / "instances_val2014.json"

        self._cat_id_to_name: Optional[Dict[int, str]] = None
        self._image_to_objects: Optional[Dict[int, Set[int]]] = None
        self._image_id_pool: Optional[List[int]] = None

    # ---------- one-time setup ----------

    def _ensure_annotations(self) -> None:
        """Download + extract the COCO 2014 annotations if missing."""
        if self.ann_path.exists():
            return

        zip_path = self.cache_dir / "annotations_trainval2014.zip"
        if not zip_path.exists():
            print(f"[COCOLoader] Downloading annotations (~250MB) -> {zip_path}")
            urlretrieve(COCO_ANN_URL, zip_path)

        print(f"[COCOLoader] Extracting instances_val2014.json")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("instances_val2014.json"):
                    zf.extract(name, self.cache_dir)
                    break

    def _load_index(self) -> None:
        """Build cat-id->name and image-id->object-ids indexes."""
        if self._cat_id_to_name is not None:
            return

        self._ensure_annotations()
        print(f"[COCOLoader] Loading annotation index from {self.ann_path}")
        with open(self.ann_path, "r") as f:
            ann = json.load(f)

        self._cat_id_to_name = {c["id"]: c["name"] for c in ann["categories"]}

        image_to_objects: Dict[int, Set[int]] = {}
        for obj in ann["annotations"]:
            image_to_objects.setdefault(obj["image_id"], set()).add(obj["category_id"])
        self._image_to_objects = image_to_objects

        # Sort image IDs so iteration is reproducible.
        self._image_id_pool = sorted(image_to_objects.keys())
        print(
            f"[COCOLoader] Index ready: {len(self._image_id_pool)} images, "
            f"{len(self._cat_id_to_name)} object categories"
        )

    # ---------- per-image fetch ----------

    def _fetch_image(self, image_id: int) -> Path:
        path = self.images_dir / f"COCO_val2014_{image_id:012d}.jpg"
        if not path.exists():
            url = COCO_IMG_URL_TEMPLATE.format(image_id)
            urlretrieve(url, path)
        return path

    def get_sample(self, image_id: int) -> COCOSample:
        self._load_index()
        path = self._fetch_image(image_id)
        gt_ids = self._image_to_objects.get(image_id, set())
        gt_names = {self._cat_id_to_name[i] for i in gt_ids}
        return COCOSample(
            image_id=image_id,
            image_path=path,
            image=Image.open(path).convert("RGB"),
            gt_objects=gt_names,
            gt_object_ids=gt_ids,
        )

    # ---------- iteration ----------

    def iter_samples(
        self,
        n: int = 100,
        start: int = 0,
        seed: Optional[int] = 42,
    ) -> Iterator[COCOSample]:
        """
        Yield up to ``n`` samples starting from offset ``start`` of the
        deterministically sorted image-id pool.

        Set ``seed`` to shuffle the pool reproducibly; ``seed=None`` keeps
        the natural sort order.
        """
        self._load_index()
        pool = list(self._image_id_pool)
        if seed is not None:
            import random
            rng = random.Random(seed)
            rng.shuffle(pool)

        end = min(start + n, len(pool))
        for image_id in pool[start:end]:
            yield self.get_sample(image_id)

    # ---------- vocabulary helpers ----------

    @property
    def categories(self) -> List[str]:
        """All 80 COCO object names."""
        self._load_index()
        return sorted(self._cat_id_to_name.values())
