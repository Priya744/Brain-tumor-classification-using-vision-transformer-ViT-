!pip install requests
import requests
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchinfo import summary
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
​
device = "cuda" if torch.cuda.is_available() else "cpu"
device

local_weights_path = "/kaggle/input/weights/vit_b_16-c867db91.pth"
pretrained_vit_weights = torch.load(local_weights_path)
pretrained_vit = torchvision.models.vit_b_16()

pretrained_vit.load_state_dict(pretrained_vit_weights)
pretrained_vit = pretrained_vit.to(device)
​

for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False
​
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
​
class ViTClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ViTClassifier, self).__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
​
    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
​
pretrained_vit.heads = nn.Identity()  # Remove the original head
model = ViTClassifier(base_model=pretrained_vit, num_classes=len(class_names)).to(device)
​
summary(model=model, 
        input_size=(32, 3, 224, 224),  # (batch_size, color_channels, height, width)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
​
train_dir = '/kaggle/input/brain-tumor-classification-mri/Training'
test_dir = '/kaggle/input/brain-tumor-classification-mri/Testing'
​
pretrained_vit_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print(pretrained_vit_transforms)
​
NUM_WORKERS = os.cpu_count()
​
def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int = NUM_WORKERS):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
​
    return train_dataloader, test_dataloader, class_names
​
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=pretrained_vit_transforms,
    batch_size=32
)
​
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
​
set_seeds()
pretrained_vit_results = train(
    model=model,
    train_dataloader=train_dataloader_pretrained,
    test_dataloader=test_dataloader_pretrained,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device
)
​
plot_loss_curves(pretrained_vit_results)
​
from going_modular.going_modular.predictions import pred_and_plot_image
​
custom_image_path = "/kaggle/input/brain-tumor-classification-mri/Testing/glioma_tumor/image(1).jpg"

pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names)
​
