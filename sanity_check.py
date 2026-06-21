import logging
from pathlib import Path
from typing import LiteralString

logger = logging.getLogger(__name__)


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

    logger.info(f"{split}: {len(seq_in)} samples checked.")


def cache_matches_tokenizer(features, tokenizer):
  """
  Verify that the cached features are compatible with the current tokenizer, 
  which may be from stale/mismatched cached features.
  """
  if not features:
    return True

  input_ids = getattr(features[0], "input_ids", None)
  if not input_ids:
    return False

  if tokenizer.cls_token_id is not None and input_ids[0] != tokenizer.cls_token_id:
    logger.warning("Cached feature CLS id %s does not match tokenizer CLS id %s",
                   input_ids[0], tokenizer.cls_token_id)
    return False

  if tokenizer.sep_token_id is not None and tokenizer.sep_token_id not in input_ids:
    logger.warning("Cached feature does not contain tokenizer SEP id %s",
                   tokenizer.sep_token_id)
    return False

  return True
