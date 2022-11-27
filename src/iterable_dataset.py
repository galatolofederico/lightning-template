import torch
import random

class IterableTemplateDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg, split):
        super(IterableTemplateDataset, self).__init__()
        samples = cfg.data[split].samples
        self.X = torch.rand((samples, cfg.data.dim))
        norm = torch.linalg.norm(self.X, dim=1)
        self.y = (norm < norm.mean()).int()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        i = random.randint(0, self.X.shape[0]-1)
        return self.X[i], self.y[i].long()

import hydra
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    ds = IterableTemplateDataset(cfg, "train")
    it = iter(ds)
    while True:
        x, y = next(it)
        print(f"x={x.shape} y={y.shape}")

if __name__ == "__main__":
    main()