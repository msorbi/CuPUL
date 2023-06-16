from transformers import RobertaForTokenClassification
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from torch import nn
import torch.nn.functional as F

class NERModel(RobertaForTokenClassification):

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = RobertaLMHead(config)
        # self.bin_classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, valid_pos):
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask)[0]
        valid_output = sequence_output[valid_pos > 0]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        softmax = nn.Softmax(dim=1)
        logits = softmax(logits)
        # logits = F.log_softmax(logits)
        # bin_logits = self.bin_classifier(sequence_output)
        return logits

    def mlm_pred(self, input_ids, attention_mask, valid_pos):
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask)[0]
        valid_output = sequence_output[valid_pos > 0]
        logits = self.lm_head(valid_output)
        return logits