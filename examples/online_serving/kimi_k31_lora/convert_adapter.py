"""Convert MindSpeed-MM LoRA safetensors names to vLLM's PEFT layout.

This converts adapter packaging only; it does not establish model support.
"""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.source.parent / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if config.get("peft_type") != "LORA" or config.get("use_dora") or config.get("use_rslora"):
        raise ValueError("Only ordinary LoRA adapters are supported")
    with safe_open(str(args.source), framework="pt", device="cpu") as source:
        names = list(source.keys())
        pairs = {}
        for name in names:
            if name.endswith(".lora_A.default.weight"):
                base, side = name.removesuffix(".lora_A.default.weight"), "A"
            elif name.endswith(".lora_B.default.weight"):
                base, side = name.removesuffix(".lora_B.default.weight"), "B"
            else:
                raise ValueError(f"Unexpected adapter parameter: {name}")
            pairs.setdefault(base, {})[side] = name
        if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
            raise ValueError("Missing or empty A/B parameter pairs")
        rank = config["r"]
        output = {}
        for base, pair in pairs.items():
            a, b = (source.get_tensor(pair[side]) for side in ("A", "B"))
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
                raise ValueError(f"Invalid A/B rank or shape: {base}")
            output[base + ".lora_A.weight"] = a.contiguous()
            output[base + ".lora_B.weight"] = b.contiguous()
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(output, str(args.output / "adapter_model.safetensors"), metadata={"format": "pt"})
    (args.output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    report = {
        "source": str(args.source),
        "source_sha256": digest(args.source),
        "output_sha256": digest(args.output / "adapter_model.safetensors"),
        "pairs": len(pairs),
        "rank": rank,
        "alpha": config["lora_alpha"],
    }
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
