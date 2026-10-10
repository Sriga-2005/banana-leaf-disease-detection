import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# MODEL COMPARISON
# ==========================================

models = [
    "Custom CNN",
    "MobileNetV2"
]

accuracy = [
    66.15,
    81.54
]

macro_precision = [
    66.95,
    79.18
]

macro_recall = [
    67.49,
    76.28
]

macro_f1 = [
    65.95,
    75.61
]

x = np.arange(len(models))
width = 0.18

plt.figure(figsize=(10, 6))

plt.bar(
    x - 1.5 * width,
    accuracy,
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    macro_precision,
    width,
    label="Macro Precision"
)

plt.bar(
    x + 0.5 * width,
    macro_recall,
    width,
    label="Macro Recall"
)

plt.bar(
    x + 1.5 * width,
    macro_f1,
    width,
    label="Macro F1"
)

plt.ylabel("Score (%)")
plt.xlabel("Model")
plt.title("Custom CNN vs MobileNetV2")

plt.xticks(x, models)
plt.ylim(0, 100)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "models/model_comparison.png",
    dpi=300
)

plt.show()