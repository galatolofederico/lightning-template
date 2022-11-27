import torch
import pytorch_lightning

class TemplateModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        *,
        input_size,
        hidden_layers,
        hidden_size,
        output_size,
        lr
    ):
        super(TemplateModel, self).__init__()
        
        self.lr = lr
        self.save_hyperparameters()

        self.net = torch.nn.Sequential()
        self.net.append(torch.nn.Linear(input_size, hidden_size))
        self.net.append(torch.nn.LeakyReLU())
        for _ in range(0, hidden_layers):
            self.net.append(torch.nn.Linear(hidden_size, hidden_size))
            self.net.append(torch.nn.LeakyReLU())
        self.net.append(torch.nn.Linear(hidden_size, output_size))

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        return self.step("train", batch)
    
    def validation_step(self, batch, it):
        return self.step("validation", batch)

    def forward(self, x):
        return self.net(x)

    def step(self, step, batch):
        X, y = batch
        logits = self(X)
        loss = self.loss_fn(logits, y)

        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == y).float().mean()
            
            self.log(f"{step}/loss", loss.item())
            self.log(f"{step}/accuracy", accuracy.item())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

import hydra
import os
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    from src.sequential_dataset import SequentialTemplateDataset
    ds = SequentialTemplateDataset(cfg,"train")
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=os.cpu_count())
    model = TemplateModel(**cfg.model, input_size=cfg.data.dim, output_size=2, lr=cfg.train.lr)

    for elem in dl:
        loss = model.training_step(elem)
        print(loss)

if __name__ == "__main__":
    main()