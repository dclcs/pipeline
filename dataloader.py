import torch
from dataset import DataSet
import torch.utils.data


def create_dataset(paths, categories, is_train):
    dataset = DataSet(paths, categories=categories, is_train=is_train)
    return dataset


class DataLoader:
    def __init__(self, paths, categories=["Chair", "Plane", "Table"], is_train=True):
        self.dataset = create_dataset(paths, categories, is_train)
        workers = 8
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=3,
            shuffle=True,
            num_workers=workers,
        )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

