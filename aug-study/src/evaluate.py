"""Оцінювання: loss/accuracy, macro-F1, confusion matrix, per-class accuracy, робастність."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torchvision.transforms import functional as TF

# Спотворення тесту (на [0,1]-тензорах). Параметри фіксовані для всіх запусків.
CORRUPTIONS = {
    "gaussian_noise": lambda x, g: (x + 0.1 * torch.randn(x.shape, generator=g).to(x.device)).clamp(0, 1),
    "brightness":     lambda x, g: (x + 0.3).clamp(0, 1),
    "contrast":       lambda x, g: ((x - x.mean(dim=(1, 2, 3), keepdim=True)) * 0.4
                                    + x.mean(dim=(1, 2, 3), keepdim=True)).clamp(0, 1),
    "rotation":       lambda x, g: torch.cat([TF.rotate(x[: len(x) // 2], 20.0),
                                              TF.rotate(x[len(x) // 2:], -20.0)]),
    "blur":           lambda x, g: TF.gaussian_blur(x, kernel_size=5, sigma=1.5),
}


@torch.no_grad()
def predict(model, loader, device, amp=False, corruption=None):
    """Повертає (loss, preds, targets). corruption - ключ із CORRUPTIONS або None."""
    model.eval()
    gen = torch.Generator().manual_seed(0)  # однакові спотворення в усіх запусках
    losses, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if corruption is not None:
            x = CORRUPTIONS[corruption](x, gen)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
        losses += F.cross_entropy(out.float(), y, reduction="sum").item()
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return losses / len(targets), preds, targets


def accuracy(preds, targets):
    return float((preds == targets).mean())


def full_test_report(model, loader, device, amp=False):
    loss, preds, targets = predict(model, loader, device, amp)
    cm = confusion_matrix(targets, preds, labels=list(range(10)))
    report = {
        "test_loss": loss,
        "test_acc": accuracy(preds, targets),
        "test_macro_f1": float(f1_score(targets, preds, average="macro")),
        "per_class_acc": (cm.diagonal() / cm.sum(axis=1)).tolist(),
    }
    robust = {}
    for name in CORRUPTIONS:
        _, p, t = predict(model, loader, device, amp, corruption=name)
        robust[name] = accuracy(p, t)
    report["robust_acc"] = robust
    report["robust_mean"] = float(np.mean(list(robust.values())))
    return report, cm
