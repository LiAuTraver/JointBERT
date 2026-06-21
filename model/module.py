import torch.nn as nn
from torch import Tensor


class IntentClassifier(nn.Module):
  def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
    super(IntentClassifier, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(input_dim, num_intent_labels)

  def forward(self, x) -> Tensor:
    x = self.dropout(x)
    return self.linear(x)


class SlotClassifier(nn.Module):
  def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
    super(SlotClassifier, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(input_dim, num_slot_labels)

  def forward(self, x) -> Tensor:
    x = self.dropout(x)
    return self.linear(x)
