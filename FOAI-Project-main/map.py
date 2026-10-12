import matplotlib.pyplot as plt
import numpy as np

# Labels for each metric
metrics = [
    "Accuracy (%)",
    "Response Time (s, lower=better)",
    "User Satisfaction (%)",
    "Context Understanding (%)",
    "Emoji/Expression Support (%)"
]

# Performance values (example data — adjust with real results)
existing_system = [70, 5.2, 60, 45, 0]
proposed_system = [92, 1.3, 90, 85, 95]

# Create bar positions
x = np.arange(len(metrics))
width = 0.35

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, existing_system, width, label="Existing System", alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, proposed_system, width, label="Proposed AI System", alpha=0.9, edgecolor='black')

# Labels and formatting
ax.set_xlabel("Performance Metrics", fontsize=12)
ax.set_ylabel("Score / Value", fontsize=12)
ax.set_title("Comparison Between Existing System and AI-based Sign2Text System", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20, ha='right')
ax.legend()

# Add values on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}", ha='center', va='bottom', fontsize=9)

# Style
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
