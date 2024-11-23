import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

import kagglehub
from Ham10000 import Ham10000 


class Loader:
    def __init__(self):
        self.path = None
        self.df_train = None
        self.df_val = None
        self.df_test = None

    def download_dataset(self):
        # Download the dataset
        self.path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

    def create_df(self):
        # Read dataset and split into train, validation, and test sets
        data_df = pd.read_csv(f"{self.path}/hmnist_28_28_RGB.csv")
        y = data_df['label']

        # Split into training (80%) and test (20%)
        self.df_train, df_test = train_test_split(data_df, test_size=0.2, random_state=42, stratify=y)
        
        # Split test into validation (50%) and test (50%)
        y = df_test['label']
        self.df_val, self.df_test = train_test_split(df_test, test_size=0.5, random_state=42, stratify=y)

        # Reset indices
        self.df_train.reset_index(drop=True, inplace=True)
        self.df_val.reset_index(drop=True, inplace=True)
        self.df_test.reset_index(drop=True, inplace=True)

        # Save the datasets to CSV files for reference
        self.df_train.to_csv("./train.csv", index=False)
        self.df_val.to_csv("./val.csv", index=False)
        self.df_test.to_csv("./test.csv", index=False)

        # Balance the training data by oversampling based on data_aug_rate
        data_aug_rate = [20, 10, 5, 50, 0, 40, 5]
        for i in range(7):
            if data_aug_rate[i]:
                self.df_train = pd.concat(
                    [self.df_train, pd.concat([self.df_train.loc[self.df_train['label'] == i, :]] * (data_aug_rate[i] - 1), ignore_index=True)],
                    ignore_index=True
                )

    def loaders(self):
        # Define data augmentation and transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Create datasets and dataloaders
        train_set = Ham10000(self.df_train, transform=train_transform)
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

        val_set = Ham10000(self.df_val, transform=val_transform)
        val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

        loaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }

        return loaders


def main():
    loader = Loader()
    loader.download_dataset()  # Download the dataset
    loader.create_df()         # Create train, validation, and test datasets
    dataloaders = loader.loaders()  # Create dataloaders


if __name__ == '__main__':
    main()
