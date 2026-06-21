import torch.nn as nn
from torch.types import Tensor
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from .module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
  def __init__(self, config, args, intent_label_lst, slot_label_lst):
    super(JointBERT, self).__init__(config)

    self.args = args
    self.num_intent_labels = len(intent_label_lst)
    self.num_slot_labels = len(slot_label_lst)
    self.bert = BertModel(config=config)  # Load pretrained bert

    self.intent_classifier = IntentClassifier(
      config.hidden_size, self.num_intent_labels, args.dropout_rate)
    self.slot_classifier = SlotClassifier(
      config.hidden_size, self.num_slot_labels, args.dropout_rate)

    if args.use_crf:
      self.crf = CRF(num_labels=self.num_slot_labels)

    if hasattr(self, "post_init"):
      self.post_init()
    else:
      self.init_weights()

  def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
    outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(input_ids, attention_mask=attention_mask,
                                                                      # sequence_output, pooled_output, (hidden_states), (attentions)
                                                                      token_type_ids=token_type_ids)
    sequence_output: Tensor = outputs[0]
    pooled_output: Tensor = outputs[1]  # [CLS]

    intent_logits: Tensor = self.intent_classifier(pooled_output)
    slot_logits: Tensor = self.slot_classifier(sequence_output)

    total_loss = 0
    # 1. Intent Softmax
    if intent_label_ids is not None:
      if self.num_intent_labels == 1:
        intent_loss_fct = nn.MSELoss()
        intent_loss = intent_loss_fct(
          intent_logits.view(-1), intent_label_ids.view(-1))
      else:
        intent_loss_fct = nn.CrossEntropyLoss()
        intent_loss = intent_loss_fct(
          intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
      total_loss += intent_loss

    # 2. Slot Softmax
    if slot_labels_ids is not None:
      if self.args.use_crf:
        slot_loss = self.crf(slot_logits, slot_labels_ids,
                             mask=attention_mask.byte())
        slot_loss = -1 * slot_loss.mean()  # negative log-likelihood
      else:
        slot_loss_fct = nn.CrossEntropyLoss(
          ignore_index=self.args.ignore_index, reduction='mean')
        # Only keep active parts of the loss
        if attention_mask is not None:
          active_loss = attention_mask.view(-1) == 1
          active_logits = slot_logits.view(-1,
                                           self.num_slot_labels)[active_loss]
          active_labels = slot_labels_ids.view(-1)[active_loss]
          slot_loss = slot_loss_fct(active_logits, active_labels)
        else:
          slot_loss = slot_loss_fct(
            slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
      total_loss += self.args.slot_loss_coef * slot_loss

    # add hidden states and attention if they are here
    outputs = ((intent_logits, slot_logits),) + outputs[2:]

    # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
    return tuple((total_loss,) + outputs)  # type: ignore
