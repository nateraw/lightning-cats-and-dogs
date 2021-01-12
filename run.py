from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10

from backbones import DummyBackbone, ResnetBackbone
from classifier import Classifier

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Backbone
    backbone = ResnetBackbone(256, 2, freeze_resnet=True)

    # Data
    data = ImageFolder(root=args.data_dir, transform=transform)
    split_size = 2500
    train_ds, val_ds = random_split(data, [len(data) - split_size, split_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    model = Classifier(backbone, learning_rate=1e-3)
    
    # Train
    trainer = pl.Trainer(max_epochs=4, gpus=1, precision=16)
    trainer.fit(model, train_loader, val_loader)
