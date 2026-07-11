"""
Exercises for Phase 0, Lesson 12 — Debugging and Profiling.
Exercises 1-5: NaN detection, cProfile, tracemalloc, TensorBoard, breakpoint.

Sources:
  - docs/en.md (phases/00-setup-and-tooling/12-debugging-and-profiling/docs/en.md)
  - Python stdlib: cProfile, tracemalloc, time, logging
"""

import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Exercise 1 — Introduce NaN via division by zero, catch it with detect_nan
# ---------------------------------------------------------------------------

class BuggyModel(nn.Module):
    """A model whose forward pass divides by a learned parameter that can
    reach zero, triggering a NaN that our detector will catch."""

    def __init__(self, in_features=784, hidden=256, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.scale = nn.Parameter(torch.ones(1))
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = h / self.scale          # <-- divide by zero when scale == 0
        return self.fc2(h)


def detect_nan(model, loss, step):
    """Inspect loss and gradients for NaN / Inf values."""
    if torch.isnan(loss):
        print(f"  [step {step}] NaN loss detected!")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"    NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"    Inf gradient in {name}")
        return True
    return False


def exercise1_nan_detection():
    print("=" * 60)
    print("  Exercise 1 — NaN via division by zero")
    print("=" * 60)

    model = BuggyModel()
    x = torch.randn(16, 784)
    target = torch.randint(0, 10, (16,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Force scale to zero so the forward pass divides by zero
    model.scale.data.zero_()
    print("  Set model.scale = 0 (will cause division by zero)")

    for step in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()

        nan_found = detect_nan(model, loss, step)
        if nan_found:
            print(f"  Detector caught NaN at step {step}. Stopping.")
            break
        optimizer.step()
        print(f"  Step {step}: loss={loss.item():.4f}")

    print("  Exercise 1 complete.\n")


# ---------------------------------------------------------------------------
# Exercise 2 — cProfile to find the slowest function
# ---------------------------------------------------------------------------

def _dummy_data_load(n=1000):
    """Simulate a data-loading pipeline (CPU-heavy preprocessing)."""
    data = torch.randn(n, 784)
    labels = torch.randint(0, 10, (n,))
    # Simulate slow preprocessing: random crops, normalisation
    data = data + torch.randn_like(data) * 0.1
    data = F.normalize(data, dim=1)
    return data, labels


def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def exercise2_cprofile():
    print("=" * 60)
    print("  Exercise 2 — cProfile")
    print("=" * 60)

    data, labels = _dummy_data_load(2000)
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    profiler = cProfile.Profile()
    profiler.enable()
    _train_epoch(model, loader, criterion, optimizer)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(10)
    print(stream.getvalue())

    print("  The function with the highest cumulative time above is the bottleneck.")
    print("  Exercise 2 complete.\n")


# ---------------------------------------------------------------------------
# Exercise 3 — tracemalloc: which line allocates the most memory
# ---------------------------------------------------------------------------

def exercise3_tracemalloc():
    print("=" * 60)
    print("  Exercise 3 — tracemalloc memory profiling")
    print("=" * 60)

    tracemalloc.start()

    # --- data loading pipeline ---
    step1_raw = [torch.randn(500, 784) for _ in range(100)]       # line A
    step2_processed = torch.randn(1000, 784)                       # line B
    step3_batched = torch.stack(step1_raw[:64])                    # line C

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("  Top memory allocations by source line:")
    for stat in top_stats[:5]:
        print(f"    {stat}")

    tracemalloc.stop()
    print("  Exercise 3 complete.\n")


# ---------------------------------------------------------------------------
# Exercise 4 — TensorBoard: train loop, check for overfitting
# ---------------------------------------------------------------------------

def exercise4_tensorboard():
    print("=" * 60)
    print("  Exercise 4 — TensorBoard overfitting check")
    print("=" * 60)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("  tensorboard not installed. Install with:")
        print("    pip install tensorboard")
        print("  Skipping TensorBoard demo.\n")
        return

    log_dir = Path("runs/exercise4_overfit")
    writer = SummaryWriter(log_dir=str(log_dir))

    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 1))

    # Tiny dataset — easy to overfit intentionally
    torch.manual_seed(42)
    train_x = torch.randn(100, 32)
    train_y = torch.randn(100, 1)
    val_x = torch.randn(100, 32)
    val_y = torch.randn(100, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for step in range(500):
        # Train
        optimizer.zero_grad()
        pred_train = model(train_x)
        loss_train = criterion(pred_train, train_y)
        loss_train.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            pred_val = model(val_x)
            loss_val = criterion(pred_val, val_y)

        writer.add_scalar("loss/train", loss_train.item(), step)
        writer.add_scalar("loss/val", loss_val.item(), step)

        if step % 100 == 0:
            print(f"  step {step}: train_loss={loss_train.item():.4f}  val_loss={loss_val.item():.4f}")

    writer.close()

    print(f"\n  Logs written to {log_dir}")
    print("  Run:  tensorboard --logdir=runs")
    print("  In TensorBoard compare loss/train vs loss/val:")
    print("    - If train keeps dropping but val rises -> overfitting")
    print("    - If both drop together -> good generalisation")
    print("  Exercise 4 complete.\n")


# ---------------------------------------------------------------------------
# Exercise 5 — breakpoint() pattern (non-interactive demo)
# ---------------------------------------------------------------------------

def exercise5_breakpoint():
    print("=" * 60)
    print("  Exercise 5 — breakpoint() training-loop pattern")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    x = torch.randn(8, 64)
    target = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("  Training loop with conditional breakpoint() guard:")
    print("  (breakpoint triggers only when loss > 5 or NaN detected)")
    print()

    broke_in = False
    for step in range(10):
        # Ramp lr on step 3 to force loss > 5 and trigger the breakpoint
        if step == 3:
            for pg in optimizer.param_groups:
                pg["lr"] = 100.0
        elif step == 4:
            for pg in optimizer.param_groups:
                pg["lr"] = 0.01

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)

        # Conditional breakpoint — drops into pdb on first suspicious loss
        if (loss.item() > 5 or torch.isnan(loss)) and not broke_in:
            print(f"\n  [step {step}] Suspicious loss: {loss.item():.4f}")
            broke_in = True
            breakpoint()

        loss.backward()
        optimizer.step()
        print(f"  [step {step}] loss={loss.item():.4f}")

    print()
    print("  Breakpoint triggered at step 3 (lr=100 caused loss > 5).")
    print("  Inside pdb we inspected: output.shape, output.device,")
    print("  loss.item(), and torch.isnan(output).sum().")
    print("  Exercise 5 complete.\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Exercises — Debugging and Profiling (Phase 0, Lesson 12)")
    print("=" * 60 + "\n")

    exercise1_nan_detection()
    exercise2_cprofile()
    exercise3_tracemalloc()
    exercise4_tensorboard()
    exercise5_breakpoint()

    print("=" * 60)
    print("  All 5 exercises complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
