import os
import datetime
import matplotlib.pyplot as plt
import re
import unicodedata
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def slugify(text: str) -> str:
	text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
	text = re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_').lower()
	return text

def save_result(title: str, description: str, fig: plt.Figure, output_dir: str = "results"):
	# 1. Timestamped directory
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	save_path = os.path.join(output_dir, timestamp)
	os.makedirs(save_path, exist_ok=True)

	# 2. Format image filename from title
	slug = slugify(title)
	img_filename = f"{slug}_plot.png" # TODO add path
	img_path = os.path.join(save_path, img_filename)
	fig.savefig(img_path)

	# 3. Write markdown report
	md_path = os.path.join(save_path, "report.md")
	with open(md_path, "w", encoding="utf-8") as f:
		f.write(f"# {title}\n")
		f.write(f"**Date**: {timestamp}\n\n")
		f.write(f"## Description\n{description}\n\n")
		f.write(f"## Visualisation\n![{title}]({img_filename})\n")

	print(f"✅ Results saved to: {save_path}")


def display_result(predicted_y, true_y ):
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	cm = confusion_matrix(true_y, predicted_y)
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non science related", "Science related"],
			yticklabels=["Non science related", "Science related"], ax=axes[0])

	axes[0].set_xlabel("Predicted Label")
	axes[0].set_ylabel("True Label")
	axes[0].set_title("Confusion Matrix")

	target_names = ['Non science related', 'Science related']
	class_report = classification_report(true_y, predicted_y, labels=[0, 1], target_names=target_names)

	axes[1].text(0, 0.5, class_report, fontsize=12, family='monospace')
	axes[1].axis("off")
	axes[1].set_title("Classification Report")

	plt.tight_layout()
	plt.show()
	return cm, class_report