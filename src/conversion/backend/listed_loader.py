import torch

class ListedLoader:
    def __init__(self, dataset, ids, batch, trace, shuffle=False, seed=2025, limit=None):
        self.dataset, self.ids, self.batch, self.trace = dataset, list(ids), batch, trace
        self.shuffle, self.seed, self.limit = shuffle, seed, limit

    def __iter__(self):
        from torch.utils.data import DataLoader, Subset
        ids = list(self.ids)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            ids = [ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()]
        if self.limit is not None: ids = ids[:self.limit * self.batch]
        used = []; self.trace.append(used)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(Subset(self.dataset, ids), batch_size=self.batch, num_workers=0,
                            shuffle=False, drop_last=False, pin_memory=True, generator=generator)
        for index, values in enumerate(loader):
            used.extend(ids[index*self.batch:index*self.batch+len(values[0])]); yield values
