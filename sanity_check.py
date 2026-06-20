from pathlib import Path
from typing import LiteralString


def check_dataset(data_dir: LiteralString | str | Path):
  data_dir = Path(data_dir)

  for split in ["train", "dev", "test"]:
    seq_in = (data_dir / split /
              "seq.in").read_text(encoding="utf-8").strip().splitlines()
    seq_out = (data_dir / split /
               "seq.out").read_text(encoding="utf-8").strip().splitlines()
    labels = (data_dir / split /
              "label").read_text(encoding="utf-8").strip().splitlines()

    assert len(seq_in) == len(seq_out) == len(
      labels), f"{split}: line number mismatch"

    for i, (x, y) in enumerate(zip(seq_in, seq_out)):
      tokens = x.split()
      slots = y.split()
      if len(tokens) != len(slots):
        raise ValueError(
            f"{split} line {i}: token-slot mismatch\n"
            f"tokens={tokens}\n"
            f"slots={slots}"
        )

    print(f"{split}: {len(seq_in)} samples checked.")
