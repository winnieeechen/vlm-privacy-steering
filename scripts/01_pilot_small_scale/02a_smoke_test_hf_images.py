from pathlib import Path
from io import BytesIO

from datasets import load_dataset
from PIL import Image

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()
OUT_DIR = ROOT / "data" / "images_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading HF dataset stream...")
ds = load_dataset(
    "dalle-mini/YFCC100M_OpenAI_subset",
    streaming=True,
    trust_remote_code=True,
)

print("Stream ready. Saving first 5 train images...")

count = 0
for item in ds["train"]:
    photoid = str(item.get("photoid", count))
    img = item.get("img")

    if isinstance(img, Image.Image):
        img.convert("RGB").save(OUT_DIR / f"{photoid}.jpg")
    elif isinstance(img, bytes):
        Image.open(BytesIO(img)).convert("RGB").save(OUT_DIR / f"{photoid}.jpg")
    elif isinstance(img, dict) and "bytes" in img:
        Image.open(BytesIO(img["bytes"])).convert("RGB").save(OUT_DIR / f"{photoid}.jpg")
    else:
        print("Unknown image type:", type(img))
        continue

    print("saved", photoid)
    count += 1

    if count >= 5:
        break

print(f"Done. Saved {count} images to {OUT_DIR}")
