import os
import datetime
import matplotlib.pyplot as plt
import re
import unicodedata

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
	img_filename = f"{slug}_plot.png"
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
