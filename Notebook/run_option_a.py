"""
Option A team runner: detect jargon spans with BERT, translate via glossary.

Usage (from the Notebook/ directory):
  python run_option_a.py "The building facade uses cantilevered structures."
  python run_option_a.py --file sentences.txt
"""

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DETECTOR_DIR = ROOT / "models" / "jargon_detector"
GLOSSARY_PATH = ROOT.parent / "Data" / "jargon_glossary.csv"

LABEL2ID = {"O": 0, "B-JARGON": 1, "I-JARGON": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class JargonGlossary:
    def __init__(self, path):
        self.by_exact = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                jargon = str(row["jargon"]).strip()
                if jargon and jargon not in self.by_exact:
                    self.by_exact[jargon] = {
                        "translation": str(row["translation"]).strip(),
                        "discipline": str(row.get("discipline", "")).strip(),
                    }
        self.by_norm = {self.normalize(k): v for k, v in self.by_exact.items()}

    @staticmethod
    def normalize(text):
        return re.sub(r"[^a-z0-9]+", "", str(text).lower())

    def lookup(self, jargon):
        key = str(jargon).strip()
        if key in self.by_exact:
            return self.by_exact[key]
        return self.by_norm.get(self.normalize(key))


def words_with_char_spans(text):
    words, spans = [], []
    for m in re.finditer(r"\S+", text):
        words.append(m.group())
        spans.append((m.start(), m.end()))
    return words, spans


def extract_spans_from_bio(words, tags):
    spans = []
    i = 0
    while i < len(words):
        if tags[i] == "B-JARGON":
            pieces = [words[i]]
            j = i + 1
            while j < len(words) and tags[j] == "I-JARGON":
                pieces.append(words[j])
                j += 1
            spans.append(" ".join(pieces))
            i = j
        else:
            i += 1
    return spans


def align_word_predictions(words, word_ids, pred_ids):
    tags = ["O"] * len(words)
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if token_idx > 0 and word_ids[token_idx - 1] == word_id:
            continue
        tags[word_id] = ID2LABEL[int(pred_ids[token_idx])]
    return tags


def translate_spans(spans, glossary):
    results = []
    for span in spans:
        cleaned = span.strip(".,;:()[]\"'")
        hit = glossary.lookup(span) or glossary.lookup(cleaned)
        if hit is None:
            results.append(
                {
                    "jargon": cleaned,
                    "translation": None,
                    "discipline": None,
                    "known": False,
                }
            )
        else:
            results.append(
                {
                    "jargon": cleaned,
                    "translation": hit["translation"],
                    "discipline": hit["discipline"],
                    "known": True,
                }
            )
    return results


def option_a_infer(sentence, glossary, model, tokenizer, device):
    s = str(sentence).strip().strip('"')
    words, _ = words_with_char_spans(s)
    if not words:
        return {"sentence": s, "predictions": [], "n_known": 0, "n_unknown": 0}

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    word_ids = encoding.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in encoding.items()}
    model.eval()
    with torch.no_grad():
        pred_ids = model(**inputs).logits.argmax(dim=-1).squeeze(0).tolist()

    tags = align_word_predictions(words, word_ids, pred_ids)
    spans = extract_spans_from_bio(words, tags)
    preds = translate_spans(spans, glossary)
    return {
        "sentence": s,
        "predictions": preds,
        "n_known": sum(1 for p in preds if p["known"]),
        "n_unknown": sum(1 for p in preds if not p["known"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Option A jargon detect + translate")
    parser.add_argument("sentence", nargs="?", help="Single sentence to analyze")
    parser.add_argument("--file", type=str, help="Text file with one sentence per line")
    args = parser.parse_args()

    if not DETECTOR_DIR.exists():
        raise SystemExit(
            f"Missing detector at {DETECTOR_DIR}. "
            "Train/save it from option_a_pipeline.ipynb Step 1 first."
        )
    if not GLOSSARY_PATH.exists():
        raise SystemExit(f"Missing glossary at {GLOSSARY_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_DIR)
    model = AutoModelForTokenClassification.from_pretrained(DETECTOR_DIR).to(device)
    glossary = JargonGlossary(GLOSSARY_PATH)

    sentences = []
    if args.file:
        sentences = [
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.sentence:
        sentences = [args.sentence]
    else:
        parser.error("Provide a sentence or --file")

    outputs = [option_a_infer(s, glossary, model, tokenizer, device) for s in sentences]
    print(json.dumps(outputs if len(outputs) > 1 else outputs[0], indent=2))


if __name__ == "__main__":
    main()
