import json
import os
from PIL import Image
from tqdm import tqdm
import argparse


def crop_symbols(json_path, images_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r") as f:
        coco = json.load(f)

    image_id_filename = {image["id"]: image["file_name"] for image in coco["images"]}

    categories = coco["categories"]
    category_id_name = {category["id"]: category["name"] for category in categories}

    count = 0
    for ann in tqdm(coco["annotations"]):
        img_id = ann["image_id"]
        category_id = ann["category_id"]
        filename = image_id_filename[img_id]

        bbox = ann["bbox"]
        x, y, w, h = [int(round(v)) for v in bbox]

        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image.width, x + w)
        y2 = min(image.height, y + h)

        crop = image.crop((x1, y1, x2, y2))
        save_name = f"symbol_{count}_{category_id_name[category_id]}.png"

        crop.save(os.path.join(out_dir, save_name))
        count += 1

    print(f"Saved {count} crops to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_path",
        type=str,
        dest="json_path",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        dest="images_dir",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        dest="out_dir",
    )

    args = parser.parse_args()

    crop_symbols(args.json_path, args.images_dir, args.out_dir)
