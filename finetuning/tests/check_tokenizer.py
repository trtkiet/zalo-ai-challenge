"""Optional real-tokenizer integration check. Downloads tokenizer files, not weights."""
import argparse
from pathlib import Path
import statistics

from transformers import AutoTokenizer
from ftkit.common import read_jsonl, write_json
from ftkit.data import encode_trajectory, trajectory, validate_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--model", default="aisingapore/Qwen-SEA-LION-v4-4B-VL")
    parser.add_argument("--revision", default="d0734aa")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    result = {"model": args.model, "revision": args.revision, "splits": {}}
    for split in ("train", "validation"):
        rows = read_jsonl(Path(args.prepared) / f"llm_{split}.jsonl")
        lengths = []
        supervised = 0
        bad_gold = []
        for record in rows:
            messages, targets = trajectory(record)
            encoded = encode_trajectory(tokenizer, messages, targets, 16384)
            if encoded is None:
                raise AssertionError(f"Record exceeds competition context: {record['id']}")
            lengths.append(len(encoded["input_ids"]))
            supervised += sum(x != -100 for x in encoded["labels"])
            for call in record["calls"]:
                if not validate_action({"tool_calls": [call]}, record["tools"]):
                    bad_gold.append({"id": record["id"], "call": call})
        result["splits"][split] = {"records": len(rows), "tokens_mean": statistics.mean(lengths),
                                   "tokens_max": max(lengths), "supervised_tokens": supervised,
                                   "over_4096": sum(x > 4096 for x in lengths), "over_8192": sum(x > 8192 for x in lengths),
                                   "gold_calls_rejected_by_schema_check": bad_gold}
    write_json(args.report, result)
    print(result)


if __name__ == "__main__":
    main()
