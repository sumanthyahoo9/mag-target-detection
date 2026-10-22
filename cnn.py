"""
Rung 3: small U-Net. 
Map in (1,H,W) -> target-centre score map out (1,H,W).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from evaluate import evaluate

# Select a device to run the CNN on
DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

def to_input(maps: np.ndarray, scale_nano_tesla: float = 10.0) -> torch.Tensor:
    """
    Remove line drift (per-column median) and scale to ~unit range. 
    Geology is left for the CNN to learn.
    A *fixed* scale (not per-patch) keeps amplitude information. 
    (N,H,W) -> (N,1,H,W).
    """
    x = (maps - np.median(maps, axis=-2, keepdims=True)) / scale_nano_tesla
    return torch.from_numpy(x[:, None].astype(np.float32))

def block(c_in: int, c_out: int) -> nn.Sequential:
    """
    A simple Convolution block made of 2D CNN layers
    """
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())


class UNet(nn.Module):
    """
    A UNet with:
        Two down/up levels. 
        Receptive field at the bottom ~ 20 m.
    Enough to see a whole anomaly with its context.
    """
    def __init__(self, c: int = 16) -> None:
        super().__init__()
        self.e1, self.e2, self.mid = block(1, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.d2, self.d1 = block(6 * c, 2 * c), block(3 * c, c)
        self.head = nn.Conv2d(c, 1, 1)
        # start by predicting "no target" (~2%)
        nn.init.constant_(self.head.bias, -4.0)   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        e1 = self.e1(x) # 100x100                                           
        e2 = self.e2(F.max_pool2d(e1, 2)) # 50x50
        m = self.mid(F.max_pool2d(e2, 2)) # 25x25
        d2 = self.d2(torch.cat([F.interpolate(m, size=e2.shape[-2:]), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:]), e1], 1))
        return self.head(d1) # logits
    
def focal_loss(logits: torch.Tensor, y: torch.Tensor, 
               gamma: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """
    CenterNet-style focal loss for Gaussian heatmaps. 
    (1-p)^gamma / p^gamma down-weighs easy pixels;
    (1-y)^beta softens the penalty for firing *near* a target. 
    Normalised by the number of targets.
    """
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = y * (1 - p) ** gamma * torch.log(p)
    neg = (1 - y) ** beta * p ** gamma * torch.log(1 - p)
    return -(pos + neg).sum() / y.sum().clamp(min=1.0)

@torch.no_grad()
def predict(model: UNet, maps: np.ndarray, batch: int = 64) -> np.ndarray:
    """
    Given a batch of maps and a model, give the scores in the maps.
    """
    model.eval()
    x = to_input(maps)
    out = [torch.sigmoid(model(x[i:i + batch].to(DEVICE))).cpu() for i in range(0, len(x), batch)]
    return torch.cat(out)[:, 0].numpy()


def train_cnn(x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, t_val: list[np.ndarray], 
              cfg: Config, epochs: int = 15, batch: int = 32, 
              lr: float = 1e-3, seed: int = 0) -> UNet:
    """
    Adam + focal loss; 
    Keep the weights with the best *validation AP* (the metric we report).
    """
    torch.manual_seed(seed)
    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = to_input(x_train), torch.from_numpy(y_train[:, None].astype(np.float32))
    best_ap, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(x)).split(batch):
            # Forward pass and loss calculation
            loss = focal_loss(model(x[idx].to(DEVICE)), y[idx].to(DEVICE))
            # Set the gradients to zero
            opt.zero_grad()
            # Backward pass
            loss.backward()
            # Update the weights
            opt.step()
        ap = evaluate(predict(model, x_val), t_val, cfg)["ap"]
        print(f"  epoch {ep + 1:2d}  last-batch loss {loss.item():.3f}  val AP {ap:.3f}")
        if ap > best_ap:
            best_ap, best_state = ap, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model
