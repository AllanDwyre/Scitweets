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

def save_result(title: str, description: str, fig: plt.Figure, output_dir: str = "results", static_dir: str = "static"):
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(static_dir, exist_ok=True)

	timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

	# 2. Format image filename from title
	slug = slugify(title)
	img_filename = f"{slug}_plot.png"
	img_path = os.path.join(static_dir, "images", img_filename)
	fig.savefig(img_path)

	# 3. Write markdown report
	md_path = os.path.join(output_dir, f"{slug}.md")
	with open(md_path, "w", encoding="utf-8") as f:
		f.write(f"# {title}\n")
		f.write(f"**Date**: {timestamp}\n\n")
		f.write(f"## Description\n{description}\n\n")
		f.write(f"## Visualisation\n![{title}]({img_path})\n")

	print(f"✅ Results saved to: {output_dir}")


def display_result(predicted_y, true_y, yticklabels = ['Non science related', 'Science related']):
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	cm = confusion_matrix(true_y, predicted_y)
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=yticklabels,
			yticklabels= yticklabels, ax=axes[0])

	axes[0].set_xlabel("Predicted Label")
	axes[0].set_ylabel("True Label")
	axes[0].set_title("Confusion Matrix")

	class_report = classification_report(true_y, predicted_y, labels=[i for i in range(len(yticklabels))], target_names=yticklabels)

	axes[1].text(0, 0.5, class_report, fontsize=12, family='monospace')
	axes[1].axis("off")
	axes[1].set_title("Classification Report")

	plt.tight_layout()
	plt.show()
	return fig