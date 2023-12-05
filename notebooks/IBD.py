import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import re
from tqdm import tqdm


def get_polygons(regions):
    polygons = []

    for region in regions.keys():
        x = regions[region]['shape_attributes']['all_points_x']
        y = regions[region]['shape_attributes']['all_points_y']
        polygon = list(zip(x, y))
        polygons.append(polygon)

    return polygons


def get_mask(mask_size, polygons):
    mask = Image.new("L", mask_size, 0)
    draw = draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        # print(polygon)
        draw.polygon(polygon, fill=255)

    mask_np = np.array(mask)
    return mask_np


class IBD(Dataset):
    def __init__(self, img_root, annotation_path):
        self.img_root = img_root
        self.annotation_path = annotation_path

        with open(annotation_path, 'r') as json_file:
            self.annotation_json = json.load(json_file)

        self.samples = self._make_dataset()
        self.length = len(self.samples)

    def _make_dataset(self):

        self.samples = []

        pattern = re.compile(r"^(.+\.png)\d+$")

        for key in tqdm(self.annotation_json.keys()):
            img_name = key[:-7][:]

            img_name = pattern.match(key).group(1)

            img_path = os.path.join(self.img_root, img_name)

            if not os.path.isfile(img_path):
                continue

            img = plt.imread(img_path)
            img = img[:, :, :3]
            mask_size = img.shape[:-1]
            regions = self.annotation_json[key]['regions']

            polygons = get_polygons(regions)
            mask = get_mask(mask_size, polygons)

            self.samples.append((img, mask))

        return self.samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    img_root = "/home/c3-0/parthpk/datasets/IB2D/train"
    annotaion_path = "/home/ma982513/Temp/akash99/annotation/via_region_data-train.json"

    dataset = IBD(img_root=img_root, annotation_path=annotaion_path)

    img, mask = dataset[1]

    print(img.shape, mask.shape, type(img), type(mask), len(dataset))


if __name__ == "__main__":
    main()

