import argparse
import os
import random
import logging
from typing import Any, Tuple, Union, List

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig, PretrainedConfig, TokenizersBackend, PreTrainedModel, BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import JointBERT, JointDistilBERT, JointAlbert

MODEL_CLASSES: dict[str, tuple[type[PretrainedConfig], type[PreTrainedModel], type[TokenizersBackend]]] = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}

PER_CLASS_SCORES = Tuple[List[float], List[float], List[float], List[int]]
AVERAGE_SCORES = Tuple[float, float, float, int]
SCORES = Union[PER_CLASS_SCORES, AVERAGE_SCORES]


def get_intent_labels(args: argparse.Namespace):
  return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args: argparse.Namespace):
  return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args: argparse.Namespace) -> TokenizersBackend:
  return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
  logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO)


def set_seed(args: argparse.Namespace):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if not args.no_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds: np.ndarray, intent_labels: np.ndarray, slot_preds: list[list], slot_labels: list[list]):
  assert len(intent_preds) == len(intent_labels) == len(
    slot_preds) == len(slot_labels)
  results: dict[str, np.ndarray | SCORES] = {}
  intent_result = get_intent_acc(intent_preds, intent_labels)
  slot_result = get_slot_metrics(slot_preds, slot_labels)
  semantic_result = get_sentence_frame_acc(
    intent_preds, intent_labels, slot_preds, slot_labels)

  results.update(intent_result)
  results.update(slot_result)
  results.update(semantic_result)

  return results


def get_slot_metrics(preds: list[list], labels: list[list]) -> dict[str, SCORES]:
  assert len(preds) == len(labels)
  return {
      "slot_precision": precision_score(labels, preds),
      "slot_recall": recall_score(labels, preds),
      "slot_f1": f1_score(labels, preds)
  }


def get_intent_acc(preds: np.ndarray, labels: np.ndarray):
  acc: np.ndarray = (preds == labels).mean()
  return {
      "intent_acc": acc
  }


def read_prediction_text(args):
  return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds: np.ndarray, intent_labels: np.ndarray, slot_preds: list[list], slot_labels: list[list]):
  """For the cases that intent and all the slots are correct (in one sentence)"""
  # Get the intent comparison result
  intent_result: np.ndarray = (intent_preds == intent_labels)

  # Get the slot comparision result
  slot_result = []
  for preds, labels in zip(slot_preds, slot_labels):
    assert len(preds) == len(labels)
    one_sent_result = True
    for p, l in zip(preds, labels):
      if p != l:
        one_sent_result = False
        break
    slot_result.append(one_sent_result)
  slot_result = np.array(slot_result)

  semantic_acc: np.ndarray = np.multiply(intent_result, slot_result).mean()
  return {
      "semantic_frame_acc": semantic_acc
  }
