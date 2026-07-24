"""Load coverage knowledge into one record per searchable chunk."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEXT_DIR = PROJECT_ROOT / "raw_text"
DEFAULT_PLANS_FILE = PROJECT_ROOT / "data" / "plans.json"
DEFAULT_PLANS_CSV = PROJECT_ROOT / "data" / "plans.csv"
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base.jsonl"
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def export_plans_to_json(
	csv_path: Path = DEFAULT_PLANS_CSV,
	json_path: Path = DEFAULT_PLANS_FILE,
) -> list[dict[str, Any]]:
	"""Export the plans CSV as a JSON array and return the exported plans."""
	with csv_path.open("r", encoding="utf-8", newline="") as plans_file:
		plans = list(csv.DictReader(plans_file))

	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")
	return plans


def _load_plans(plans_path: Path) -> list[dict[str, Any]]:
	"""Read plans from JSON, creating the export from CSV when needed."""
	if not plans_path.exists():
		export_plans_to_json(json_path=plans_path)

	with plans_path.open("r", encoding="utf-8") as plans_file:
		plans = json.load(plans_file)

	if not isinstance(plans, list):
		raise ValueError(f"Expected a JSON array in {plans_path}")
	return plans


def _policy_section(text_path: Path, text_chunk: str) -> str:
	"""Map a policy chunk to the most useful coverage section."""
	text_lower = text_chunk.lower()
	for section in ("exclusions", "claims", "enrollment", "coverage"):
		if section in text_lower:
			return section

	file_name = text_path.stem.lower()
	if "claim" in file_name:
		return "claims"
	if "enroll" in file_name:
		return "enrollment"
	return "coverage"


def load_documents(
	text_dir: Path = DEFAULT_TEXT_DIR,
	plans_path: Path = DEFAULT_PLANS_FILE,
) -> list[dict[str, Any]]:
	"""Load all text files and one natural-language chunk for each plan."""
	documents: list[dict[str, Any]] = []
	ingested_at = datetime.now(timezone.utc).isoformat()

	for text_path in sorted(text_dir.glob("*.txt")):
		text_chunks = TEXT_SPLITTER.split_text(text_path.read_text(encoding="utf-8"))
		for chunk_index, text_chunk in enumerate(text_chunks):
			documents.append(
				{
					"id": f"text:{text_path.stem}:{chunk_index}",
					"text": text_chunk,
					"source_file": str(text_path),
					"source_type": "unstructured",
					"plan_type": None,
					"section": _policy_section(text_path, text_chunk),
					"ingested_at": ingested_at,
				}
			)

	for plan in _load_plans(plans_path):
		documents.append(
			{
				"id": f"plan:{plan['plan_id']}",
				"text": (
					f"{plan['plan_name']}: ${plan['monthly_premium']}/month premium, "
					f"${plan['annual_deductible']} deductible, "
					f"{plan['copay_pct']}% coinsurance, "
					f"network: {plan['network_tier']}"
				),
				"source_file": str(plans_path),
				"source_type": "structured",
				"plan_type": plan["coverage_type"],
				"section": "coverage",
				"ingested_at": ingested_at,
			}
		)

	return documents


def write_knowledge_base(
	documents: list[dict[str, Any]],
	output_path: Path = DEFAULT_KNOWLEDGE_BASE,
) -> None:
	"""Write one JSON object per line to the knowledge-base JSONL file."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as knowledge_base_file:
		for document in documents:
			knowledge_base_file.write(json.dumps(document) + "\n")


def sanity_check(
	documents: list[dict[str, Any]], sample_size: int = 5,
) -> list[dict[str, Any]]:
	"""Print the count and a random sample for manual coherence review."""
	sample = random.sample(documents, min(sample_size, len(documents)))
	print(f"Total chunk count: {len(documents)}")
	print(f"Reading {len(sample)} random chunks:")
	for document in sample:
		print(f"\n[{document['id']}] {document['text']}")
	return sample


if __name__ == "__main__":
	documents = load_documents()
	write_knowledge_base(documents)
	sanity_check(documents)
