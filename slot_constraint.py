import argparse
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

MASKED_LOGIT_VALUE = -1e9


def _read_lines(path: str) -> list[str]:
  with open(path, "r", encoding="utf-8") as f:
    return [line.strip() for line in f]


def _add_slot_with_counterpart(allowed: np.ndarray, slot_label: str, slot_to_id: dict[str, int]):
  slot_id = slot_to_id.get(slot_label)
  if slot_id is not None:
    allowed[slot_id] = True

  if slot_label.startswith("B-"):
    counterpart = "I-" + slot_label[2:]
  elif slot_label.startswith("I-"):
    counterpart = "B-" + slot_label[2:]
  else:
    return

  counterpart_id = slot_to_id.get(counterpart)
  if counterpart_id is not None:
    allowed[counterpart_id] = True


def build_intent_slot_mask(args: argparse.Namespace,
                           intent_label_lst: list[str],
                           slot_label_lst: list[str]) -> np.ndarray | None:
  """Build a mask of slot labels observed with each intent in the training set."""
  train_dir = os.path.join(args.data_dir, args.task, "train")
  intent_path = os.path.join(train_dir, "label")
  slot_path = os.path.join(train_dir, "seq.out")

  if not os.path.exists(intent_path) or not os.path.exists(slot_path):
    logger.warning("Intent-slot constraints disabled because training labels were not found in %s", train_dir)
    return None

  intent_to_id = {label: idx for idx, label in enumerate(intent_label_lst)}
  slot_to_id = {label: idx for idx, label in enumerate(slot_label_lst)}
  mask = np.zeros((len(intent_label_lst), len(slot_label_lst)), dtype=bool)
  seen_intents: set[int] = set()

  default_allowed = np.ones(len(slot_label_lst), dtype=bool)
  for special_label in {getattr(args, "slot_pad_label", "PAD"), "PAD", "UNK"}:
    special_id = slot_to_id.get(special_label)
    if special_id is not None:
      default_allowed[special_id] = False

  o_id = slot_to_id.get("O")
  if o_id is not None:
    mask[:, o_id] = True
    default_allowed[o_id] = True

  intent_lines = _read_lines(intent_path)
  slot_lines = _read_lines(slot_path)
  for intent_label, slot_line in zip(intent_lines, slot_lines):
    intent_id = intent_to_id.get(intent_label)
    if intent_id is None:
      intent_id = intent_to_id.get("UNK")
    if intent_id is None:
      continue

    seen_intents.add(intent_id)
    for slot_label in slot_line.split():
      _add_slot_with_counterpart(mask[intent_id], slot_label, slot_to_id)

  intent_parts = [set(label.split("#")) for label in intent_label_lst]
  for intent_id, parts in enumerate(intent_parts):
    if len(parts) == 1:
      continue
    for other_intent_id, other_parts in enumerate(intent_parts):
      if intent_id != other_intent_id and parts.intersection(other_parts):
        mask[intent_id] = np.logical_or(mask[intent_id], mask[other_intent_id])
        mask[other_intent_id] = np.logical_or(mask[other_intent_id], mask[intent_id])

  for intent_id in range(len(intent_label_lst)):
    if intent_id not in seen_intents:
      mask[intent_id] = default_allowed

  logger.info("Intent-slot constraint mask built from %s", train_dir)
  return mask


def get_intent_predictions_and_confidences(intent_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  intent_preds = np.argmax(intent_logits, axis=1)
  shifted_logits = intent_logits - np.max(intent_logits, axis=1, keepdims=True)
  exp_logits = np.exp(shifted_logits)
  intent_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  confidences = intent_probs[np.arange(intent_logits.shape[0]), intent_preds]
  return intent_preds, confidences


def decode_slot_predictions(intent_logits: np.ndarray,
                            slot_logits: np.ndarray,
                            intent_slot_mask: np.ndarray | None,
                            threshold: float) -> tuple[np.ndarray, int]:
  """Decode slot logits with optional intent-aware label masking."""
  if intent_slot_mask is None:
    return np.argmax(slot_logits, axis=2), 0

  intent_preds, confidences = get_intent_predictions_and_confidences(intent_logits)
  masked_slot_logits = slot_logits.copy()
  applied_count = 0

  for example_idx, intent_id in enumerate(intent_preds):
    if confidences[example_idx] < threshold:
      continue

    allowed_mask = intent_slot_mask[intent_id]
    if not np.any(allowed_mask):
      continue

    masked_slot_logits[example_idx, :, ~allowed_mask] = MASKED_LOGIT_VALUE
    applied_count += 1

  return np.argmax(masked_slot_logits, axis=2), applied_count


def repair_bio_slot_sequence(slot_sequence: list[str], slot_label_lst: list[str]) -> list[str]:
  slot_label_set = set(slot_label_lst)
  repaired = []
  previous_entity = None

  for label in slot_sequence:
    current_label = label
    if label.startswith("I-"):
      entity = label[2:]
      if previous_entity != entity:
        begin_label = "B-" + entity
        if begin_label in slot_label_set:
          current_label = begin_label

    repaired.append(current_label)

    if current_label.startswith("B-") or current_label.startswith("I-"):
      previous_entity = current_label[2:]
    else:
      previous_entity = None

  return repaired