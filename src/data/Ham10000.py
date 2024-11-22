import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import kagglehub
import pandas as pd

class Ham10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

def data_loader():

    # Download the dataset
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    
    # Read dataset and split into train, validation, and test sets
    data_df = pd.read_csv(f"{path}/hmnist_28_28_RGB.csv")
    y = data_df['label']

    # Split into training (80%) and test (20%)
    df_train, df_test = train_test_split(data_df, test_size=0.2, random_state=42, stratify=y)
    
    # Split test into validation (50%) and test (50%)
    y = df_test['label']
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42, stratify=y)

    # Reset indices
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Save the datasets to CSV files for reference
    df_train.to_csv("./train.csv", index=False)
    df_val.to_csv("./val.csv", index=False)
    df_test.to_csv("./test.csv", index=False)

    # Balance the training data by oversampling based on data_aug_rate
    data_aug_rate = [20, 10, 5, 50, 0, 40, 5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train = pd.concat(
                [df_train, pd.concat([df_train.loc[df_train['label'] == i, :]] * (data_aug_rate[i] - 1), ignore_index=True)],
                ignore_index=True
            )

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
    train_set = Ham10000(df_train, transform=train_transform)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    val_set = Ham10000(df_val, transform=val_transform)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    loaders = {
        'train': train_dataloader,
        'val': val_dataloader
    }
    print(loaders)
    return loaders

def main():
    loader = data_loader()


if __name__ == '__main__':
    main()
