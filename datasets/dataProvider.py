from torch.utils.data import DataLoader, ConcatDataset
from datasets.utils import FullDatasetBase

__all__ = ["DataProvider", "DatasetFactory"]


class DataProvider:
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader

    def __init__(self, params: dict):
        self.factory = DatasetFactory()
        self.dataset, self.dataset_params, train, val, test = self.factory.build_dataset(params)

        if 'workers' in params:
            workers = params['workers']
            params.pop('workers')
        else:
            workers = 4
        if 'adv_dataset' in self.dataset_params and self.dataset_params['adv_dataset']:
            print("Since using adv dataset which needs cuda, so using dataloader in main process with worker=0.")
            workers = 0

        train_bz = val_bz = test_bz = params['batch_size']
        params.pop('batch_size')
        if 'train_bz' in params:
            train_bz = params['train_bz']
        if 'test_bz' in params:
            val_bz = test_bz = params['test_bz']

        self.train_dl = self._create_dataloader(train, shuffle=True, workers=workers, batch_size=train_bz, **params)
        if 'repeat' in params:
            params.pop('repeat')
        self.val_dl = self._create_dataloader(val, shuffle=False, workers=workers, batch_size=val_bz, **params)
        self.test_dl = self._create_dataloader(test, shuffle=False, workers=workers, batch_size=test_bz, **params)

    @staticmethod
    def _create_dataloader(base_dataset, workers=4, batch_size=256, drop_last=False, shuffle=False, repeat=1,
                           collate_fn=None, **kwargs):
        if repeat > 1:
            base_dataset = ConcatDataset([base_dataset] * repeat)
        loader = DataLoader(base_dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
                            drop_last=drop_last, pin_memory=True, collate_fn=collate_fn)
        print(f"len(dataset): {len(base_dataset)}")
        return loader


class DatasetFactory:
    from datasets.super_resolution import SRFullDataset

    dataset_params: dict
    all_datasets = [SRFullDataset]

    @staticmethod
    def build_dataset(dataset_params):
        dataset_name = dataset_params['name']
        assert isinstance(dataset_name, str)

        dataset_type, dataset_params = DatasetFactory.analyze_name(dataset_name, dataset_params)
        print("dataset_params:", dataset_params)
        dataset = dataset_type(**dataset_params)
        train, val, test = DatasetFactory.gen_datasets(dataset, dataset_params)
        return dataset, dataset_params, train, val, test

    @staticmethod
    def analyze_name(name: str, params, type_only=False):
        import re

        if "dataset_mapping" not in params:
            params["dataset_mapping"] = (0, 1, 2)
        if "dataset_transforms" not in params:  # 0-> train transforms 1-> test transforms
            params["dataset_transforms"] = (0, 1, 1)

        dataset_type = None
        for d in DatasetFactory.all_datasets:
            if d.is_dataset_name(name):
                dataset_type = d
                break

        if dataset_type is None:
            raise NotImplementedError("Dataset Not Implemented")

        if type_only:
            return dataset_type
        else:
            return dataset_type, params

    @staticmethod
    def gen_datasets(dataset: FullDatasetBase, params):
        trans = [dataset.gen_train_transforms(), dataset.gen_test_transforms()]
        data_gens: list = [dataset.gen_train_datasets, dataset.gen_val_datasets, dataset.gen_test_datasets]

        train, val, test = [data_gens[params['dataset_mapping'][i]](*trans[params['dataset_transforms'][i]]) for i in
                            [0, 1, 2]]

        return train, val, test
