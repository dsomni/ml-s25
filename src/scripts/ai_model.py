import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import (
    RobertaConfig, RobertaModel, DebertaV2Config, DebertaV2Model
)


# Similar to PyTorch's native torch.nn.MarginRankingLoss but with the sigmoid operation on arguments
# Check https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
def get_ranking_loss(logits, labels, margin=0.7):
    logits = torch.sigmoid(logits)
    labels1 = labels.unsqueeze(1)
    labels2 = labels.unsqueeze(0)

    logits1 = logits.unsqueeze(1)
    logits2 = logits.unsqueeze(0)

    y_ij = torch.sign(labels1 - labels2)
    r_ij = logits1 - logits2

    loss = torch.clamp(-r_ij * y_ij + margin, min=0.0).mean()
    return loss


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Rank Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ModelType:
    ROBERTA = 1,
    DEBERTA = 2,

class AiModel(nn.Module):
    """
    The LLM Detect AI Generated Text Model
    """

    def __init__(self, device, model_name, model_type):
        print("initializing the Rank Model...")

        super(AiModel, self).__init__()
        # ----------------------------- Backbone -----------------------------------------#
        if model_type == ModelType.ROBERTA:
            backbone_config = RobertaConfig.from_pretrained(model_name)
        elif model_type == ModelType.DEBERTA:
            backbone_config = DebertaV2Config.from_pretrained(model_name)
        else:
            raise NotImplementedError

        backbone_config.update(
            {
                "use_cache": False,
            }
        )

        self.backbone = None

        if model_type == ModelType.ROBERTA:
            self.backbone = RobertaModel.from_pretrained(
                model_name, config=backbone_config
            )
        elif model_type == ModelType.DEBERTA:
            self.backbone = DebertaV2Model.from_pretrained(
                model_name, config=backbone_config
            )
        else:
            raise NotImplementedError

        self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(0.05)

        # classifier
        num_features = self.backbone.config.hidden_size
        self.classifier = nn.Linear(num_features, 1)

        self.pool = MeanPooling()

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        encoder_layer = outputs.last_hidden_state
        embeddings = self.pool(encoder_layer, attention_mask)

        return embeddings

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # features
        features = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        features = self.dropout(features)
        logits = self.classifier(features).reshape(-1)

        # loss
        loss = None
        if labels is not None:
            labels = labels.reshape(-1)
            loss = get_ranking_loss(logits, labels)

        return logits, loss
