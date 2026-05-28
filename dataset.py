import os
import glob
import json
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class StanfordDogs(Dataset):
    def __init__(
        self,
        root,
        synset_to_imagenet_idx,
        transform=None
    ):

        self.root = root
        self.synset_to_imagenet_idx = synset_to_imagenet_idx

        self.transform = transform or transforms.ToTensor()

        image_root = os.path.join(root, 'Images')

        all_images = glob.glob(
            os.path.join(image_root, '*', '*.jpg')
        )

        self.sample_list = []

        for full_img_path in all_images:

            rel_path = os.path.relpath(
                full_img_path,
                start=image_root
            )

            self.sample_list.append(rel_path)

        print(f'Loaded {len(self.sample_list)} samples.')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_rel_path = self.sample_list[idx]

        img_path = os.path.join(
            self.root,
            'Images',
            img_rel_path
        )

        image = Image.open(img_path).convert('RGB')

        orig_w, orig_h = image.size

        breed, fname = os.path.split(img_rel_path)

        xml_path = os.path.join(
            self.root,
            'Annotation',
            breed,
            os.path.splitext(fname)[0]
        )

        tree = ET.parse(xml_path)

        objects = tree.findall('object')

        bboxes = []

        for obj in objects:

            box = obj.find('bndbox')

            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)

            bboxes.append([xmin, ymin, xmax, ymax])

        # Keep largest bounding box
        if not bboxes:
            bbox = [0, 0, orig_w, orig_h]
        else:
            bbox = max(
                bboxes,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )

        synset = breed.split('-')[0]

        imagenet_class = self.synset_to_imagenet_idx[synset]

        image = self.transform(image)

        return (
            image,
            torch.tensor(bbox, dtype=torch.float32),
            imagenet_class,
            [orig_w, orig_h]
        )


def load_synset_mapping(
    path_to_map_clsloc='imagenet_class_index.json'
):

    with open(path_to_map_clsloc, 'r') as f:
        class_index = json.load(f)
    
    synsets = {
        class_index[str(i)][0]: i
        for i in range(1000)
    }

    return synsets


if __name__ == '__main__':

    ROOT = r'SDOGS_SMALL'

    MAP_CLSLOC_PATH = 'imagenet_class_index.json'

    IMAGE_SIZE = 224

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = StanfordDogs(
        root=ROOT,
        synset_to_imagenet_idx=load_synset_mapping(MAP_CLSLOC_PATH),
        transform=transform
    )

    print(len(dataset))

    image, bbox, cls, size = dataset[0]

    print(image.shape)
    print(bbox)
    print(cls)
    print(size)