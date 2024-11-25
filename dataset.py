from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset


def get_loader(root1: Path, root2: Path, img_size: Tuple[int]=(512, 512), batch_size:int=1):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    return DataLoader(CustomDataset(root1, root2, tf),
                      batch_size=batch_size,
                      shuffle=True)
    
class CustomDataset(VisionDataset):
    def __init__(self, root1, root2, transform=None):
        super().__init__(root1, transform=transform)

        self.files1 = list(sorted(root1.glob('*.png'))) + list(sorted(root1.glob('*.jpg')))
        self.files2 = list(sorted(root2.glob('*.png'))) + list(sorted(root2.glob('*.jpg')))


    def __len__(self):
        return min(len(self.files1), len(self.files2), 1000)

    def __getitem__(self, idx):
        img1 = self.transform(Image.open(self.files1[idx]).convert('RGB'))
        img2 = self.transform(Image.open(self.files2[idx]).convert('RGB'))
        return img1, img2



    