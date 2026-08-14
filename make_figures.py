"""Regenerate the report's three figures from the frozen experiment artefacts.

Read-only over results/: consumes results/*.csv and *_history.json plus the
kagglehub CIFAKE cache, writes print-quality PDFs into report/figures/.
No training, no mutation of results/.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.io import decode_jpeg, encode_jpeg

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

#IEEEtran geometry: 3.5 in column, 7.16 in text width.
COLUMN_W = 3.5
TEXT_W = 7.16

plt.rcParams.update({
  "font.size": 8,
  "axes.titlesize": 8,
  "axes.labelsize": 8,
  "legend.fontsize": 7,
  "xtick.labelsize": 7,
  "ytick.labelsize": 7,
  "figure.dpi": 300,
  "savefig.bbox": "tight",
})

#Same severity lists as the notebook; verified against Hendrycks & Dietterich's
#make_cifar_c.py 32x32 values (lines 131, 159, 364 of the upstream file).
SEVERITIES = {
  "noise": [0.04, 0.06, 0.08, 0.09, 0.10],
  "blur": [0.4, 0.6, 0.7, 0.8, 1.0],
  "jpeg": [80, 65, 58, 50, 40],
}


def corrupt(images: torch.Tensor, kind: str, severity: float) -> torch.Tensor:
  """The notebook's corrupt(), unchanged, so Fig. 1 shows the real pipeline."""
  if kind == "none" or severity is None:
    return images
  if kind == "blur":
    k = int(2 * np.ceil(3 * severity) + 1)
    return TF.gaussian_blur(images, kernel_size=[k, k], sigma=[severity, severity])
  if kind == "noise":
    return (images + torch.randn_like(images) * severity).clamp(0.0, 1.0)
  if kind == "jpeg":
    as_uint8 = (images.clamp(0, 1) * 255).to(torch.uint8)
    out = torch.stack([
      decode_jpeg(encode_jpeg(img, quality=int(severity))) for img in as_uint8
    ])
    return out.float() / 255.0
  raise ValueError(f"Unknown corruption kind: {kind}")


def load_results() -> pd.DataFrame:
  cnn = pd.read_csv(RESULTS_DIR / "all_results.csv")
  resnet = pd.read_csv(RESULTS_DIR / "resnet18_all_results.csv")
  df = pd.concat([cnn, resnet], ignore_index=True)
  assert len(df) == 192, f"expected 192 rows (2 files x 96), got {len(df)}"
  return df


def fig1_corruption_examples() -> None:
  """One test image at the report's headline conditions."""
  import kagglehub

  data_root = kagglehub.dataset_download(
    "birdy654/cifake-real-and-ai-generated-synthetic-images")
  test_set = torchvision.datasets.ImageFolder(
    os.path.join(data_root, "test"),
    transform=torchvision.transforms.ToTensor())
  torch.manual_seed(42)  #fixes the noise draw so the figure reproduces exactly
  image = test_set[10_000][0]  #first REAL image; index is stable (sorted files)

  conditions = [
    ("none", None, "clean"),
    ("noise", 0.10, r"noise $\sigma$=0.10"),
    ("jpeg", 40, "JPEG q=40"),
    ("blur", 0.6, r"blur $\sigma$=0.6"),
    ("blur", 1.0, r"blur $\sigma$=1.0"),
  ]
  fig, axes = plt.subplots(1, 5, figsize=(COLUMN_W, 0.95))
  for ax, (kind, sev, title) in zip(axes, conditions):
    out = corrupt(image.unsqueeze(0), kind, sev)[0]
    ax.imshow(out.permute(1, 2, 0).numpy())
    ax.set_title(title, fontsize=6)
    ax.axis("off")
  fig.savefig(FIG_DIR / "fig1_corruptions.pdf")
  plt.close(fig)


def _plot_accuracy_vs_severity(df: pd.DataFrame, axes) -> None:
  """Plot one corruption family per axis with +/-1 SD bands."""
  stats = (df.groupby(["model", "strategy", "corruption", "severity"])
             ["accuracy"].agg(["mean", "std"]).reset_index())
  styles = {
    ("BaselineCNN", "baseline"): dict(color="tab:blue", ls="--", label="CNN baseline"),
    ("BaselineCNN", "robust"): dict(color="tab:blue", ls="-", label="CNN robust"),
    ("ResNet18", "baseline"): dict(color="tab:red", ls="--", label="ResNet-18 baseline"),
    ("ResNet18", "robust"): dict(color="tab:red", ls="-", label="ResNet-18 robust"),
  }
  titles = {
    "noise": r"Gaussian noise (trained on)",
    "jpeg": "JPEG compression (trained on)",
    "blur": r"Gaussian blur (held out)",
  }
  xlabels = {"noise": r"noise $\sigma$", "jpeg": "JPEG quality", "blur": r"blur $\sigma$"}

  for ax, kind in zip(axes, ["noise", "jpeg", "blur"]):
    for (model, strategy), style in styles.items():
      sub = (stats[(stats["model"] == model) & (stats["strategy"] == strategy)
                   & (stats["corruption"] == kind)]
             .set_index("severity").loc[[float(s) for s in SEVERITIES[kind]]])
      ax.plot(sub.index, sub["mean"], marker="o", ms=2.5, lw=1, **style)
      ax.fill_between(sub.index, sub["mean"] - sub["std"],
                      sub["mean"] + sub["std"], color=style["color"], alpha=0.15)
      clean = stats[(stats["model"] == model) & (stats["strategy"] == strategy)
                    & (stats["corruption"] == "none")]["mean"].item()
      ax.axhline(clean, color=style["color"], ls=":", lw=0.6, alpha=0.6)
    ax.set_title(titles[kind])
    ax.set_xlabel(xlabels[kind])
    ax.set_ylabel("test accuracy")
    if kind == "jpeg":
      ax.invert_xaxis()  #lower quality = worse, so severity still grows rightwards
  axes[0].legend(loc="lower left", frameon=False)


def fig2_accuracy_vs_severity(df: pd.DataFrame) -> None:
  """Original side-by-side accuracy panels for the LaTeX report."""
  fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.1), sharey=True)
  _plot_accuracy_vs_severity(df, axes)
  fig.savefig(FIG_DIR / "fig2_accuracy_severity.pdf")
  plt.close(fig)


def fig2_accuracy_vs_severity_stacked(df: pd.DataFrame) -> None:
  """Word-port variant with one full-width corruption panel per row."""
  fig, axes = plt.subplots(3, 1, figsize=(COLUMN_W, 5.6), sharey=True)
  fig.subplots_adjust(hspace=0.65)
  _plot_accuracy_vs_severity(df, axes)
  fig.savefig(FIG_DIR / "fig2_accuracy_severity_stacked.pdf")
  plt.close(fig)


def _plot_loss_curves(axes) -> None:
  """Plot seed-42 histories into one axis per architecture."""
  runs = {
    "BaselineCNN": [("baseline_seed42", "baseline"), ("robust_seed42", "robust")],
    "ResNet-18": [("resnet18_baseline_seed42", "baseline"),
                  ("resnet18_robust_seed42", "robust")],
  }
  colors = {"baseline": "tab:blue", "robust": "tab:orange"}
  for ax, (model, model_runs) in zip(axes, runs.items()):
    for stem, strategy in model_runs:
      with open(RESULTS_DIR / f"{stem}_history.json") as f:
        h = json.load(f)
      epochs = range(1, h["epochs_run"] + 1)
      ax.plot(epochs, h["train_losses"], color=colors[strategy], ls="--",
              lw=0.9, label=f"{strategy} train")
      ax.plot(epochs, h["val_losses"], color=colors[strategy], ls="-",
              lw=0.9, label=f"{strategy} val")
      ax.axvline(h["best_epoch"], color=colors[strategy], ls=":", lw=0.6, alpha=0.7)
    ax.set_title(model)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(frameon=False, fontsize=5.5)


def fig3_loss_curves() -> None:
  """Original side-by-side training curves for the LaTeX report."""
  fig, axes = plt.subplots(1, 2, figsize=(COLUMN_W, 1.7), sharey=False)
  #wspace clears the right panel's y-tick labels out of the inter-panel gap
  fig.subplots_adjust(wspace=0.42)
  _plot_loss_curves(axes)
  fig.savefig(FIG_DIR / "fig3_loss_curves.pdf")
  plt.close(fig)


def fig3_loss_curves_stacked() -> None:
  """Word-port variant with one full-width architecture panel per row."""
  fig, axes = plt.subplots(2, 1, figsize=(COLUMN_W, 3.6), sharey=False)
  fig.subplots_adjust(hspace=0.55)
  _plot_loss_curves(axes)
  fig.savefig(FIG_DIR / "fig3_loss_curves_stacked.pdf")
  plt.close(fig)


if __name__ == "__main__":
  df = load_results()
  fig1_corruption_examples()
  fig2_accuracy_vs_severity(df)
  fig2_accuracy_vs_severity_stacked(df)
  fig3_loss_curves()
  fig3_loss_curves_stacked()
  for name in ["fig1_corruptions", "fig2_accuracy_severity",
               "fig2_accuracy_severity_stacked", "fig3_loss_curves",
               "fig3_loss_curves_stacked"]:
    path = FIG_DIR / f"{name}.pdf"
    assert path.exists() and path.stat().st_size > 0, f"missing {name}"
    print(f"{name}.pdf  {path.stat().st_size} bytes")
