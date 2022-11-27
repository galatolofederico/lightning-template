import torch

class SequentialTemplateDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super(SequentialTemplateDataset, self).__init__()
        samples = cfg.data[split].samples
        self.X = torch.rand((samples, cfg.data.dim))
        norm = torch.linalg.norm(self.X, dim=1)
        self.y = (norm < norm.mean()).int()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i].long()

import hydra
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    ds = SequentialTemplateDataset(cfg, "train")
    
    for x, y in ds:
        print(f"x={x.shape} y={y.shape}")

if __name__ == "__main__":
    main()