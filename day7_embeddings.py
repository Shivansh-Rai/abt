"""Embed the knowledge base and plot its chunks in two dimensions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base.jsonl"
EMBEDDED_KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base_embeddings.jsonl"
PLOT_PATH = PROJECT_ROOT / "embeddings_2d.png"
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
	global _model
	if _model is None:
		_model = SentenceTransformer(MODEL_NAME)
	return _model


def embed(text: str) -> list[float]:
	"""Return an all-MiniLM-L6-v2 embedding for one text string."""
	return _get_model().encode(text, convert_to_numpy=True).tolist()


def load_chunks(path: Path = KNOWLEDGE_BASE_PATH) -> list[dict[str, Any]]:
	"""Read one JSON object per line from the knowledge base."""
	with path.open("r", encoding="utf-8") as knowledge_base_file:
		return [json.loads(line) for line in knowledge_base_file if line.strip()]


def embed_chunks(
	chunks: list[dict[str, Any]],
	output_path: Path = EMBEDDED_KNOWLEDGE_BASE_PATH,
) -> list[dict[str, Any]]:
	"""Embed every chunk and save the vectors alongside their source records."""
	for chunk in chunks:
		chunk["embedding"] = embed(chunk["text"])

	with output_path.open("w", encoding="utf-8") as embedded_file:
		for chunk in chunks:
			embedded_file.write(json.dumps(chunk) + "\n")
	return chunks


def plot_embeddings(
	chunks: list[dict[str, Any]],
	output_path: Path = PLOT_PATH,
) -> None:
	"""Reduce embeddings to two dimensions and save a section-colored scatter plot."""
	if len(chunks) < 2:
		raise ValueError("At least two chunks are required for PCA")

	embeddings = [chunk["embedding"] for chunk in chunks]
	coordinates = PCA(n_components=2).fit_transform(embeddings)
	colors = {
		"coverage": "#2563eb",
		"exclusions": "#dc2626",
		"claims": "#16a34a",
		"enrollment": "#d97706",
	}

	figure, axis = plt.subplots(figsize=(9, 6))
	for section, color in colors.items():
		section_coordinates = [
			coordinate
			for coordinate, chunk in zip(coordinates, chunks)
			if chunk["section"] == section
		]
		if section_coordinates:
			axis.scatter(
				[coordinate[0] for coordinate in section_coordinates],
				[coordinate[1] for coordinate in section_coordinates],
				c=color,
				label=section,
				s=70,
				alpha=0.85,
			)

	axis.set_title("Knowledge Base Embeddings (PCA)")
	axis.set_xlabel("Principal component 1")
	axis.set_ylabel("Principal component 2")
	axis.legend(title="Section")
	axis.grid(alpha=0.2)
	figure.tight_layout()
	figure.savefig(output_path, dpi=160)
	plt.close(figure)


def main() -> None:
	chunks = embed_chunks(load_chunks())
	plot_embeddings(chunks)
	print(f"Embedded {len(chunks)} chunks")
	print(f"Saved embeddings to {EMBEDDED_KNOWLEDGE_BASE_PATH}")
	print(f"Saved PCA plot to {PLOT_PATH}")


if __name__ == "__main__":
	main()