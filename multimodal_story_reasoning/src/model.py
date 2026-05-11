import torch
import torch.nn as nn
import timm

from transformers import DistilBertModel


class VisualEncoder(nn.Module):

    def __init__(self, output_dim=256):

        super(VisualEncoder, self).__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        self.fc = nn.Linear(1280, output_dim)

    def forward(self, images):

        features = self.backbone(images)

        features = self.fc(features)

        return features


class TextEncoder(nn.Module):

    def __init__(self, output_dim=256):

        super(TextEncoder, self).__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        cls_embedding = self.fc(cls_embedding)

        return cls_embedding


class CrossModalAttention(nn.Module):

    def __init__(self, feature_dim=256):

        super(CrossModalAttention, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, visual_features, text_features):

        visual_features = visual_features.unsqueeze(1)

        text_features = text_features.unsqueeze(1)

        attended_features, attention_weights = self.attention(
            query=visual_features,
            key=text_features,
            value=text_features
        )

        attended_features = attended_features.squeeze(1)

        return attended_features, attention_weights


class SequenceModel(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=256):

        super(SequenceModel, self).__init__()

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sequence_features):

        outputs, hidden = self.bigru(sequence_features)

        return outputs


class TextDecoder(nn.Module):

    def __init__(self, input_dim=512, vocab_dim=30522):

        super(TextDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, vocab_dim)

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


class MultimodalStoryModel(nn.Module):

    def __init__(self):

        super(MultimodalStoryModel, self).__init__()

        self.visual_encoder = VisualEncoder()

        self.text_encoder = TextEncoder()

        self.cross_attention = CrossModalAttention()

        self.sequence_model = SequenceModel()

        self.decoder = TextDecoder()

    def forward(
        self,
        images,
        input_ids,
        attention_mask
    ):

        visual_features = self.visual_encoder(images)

        text_features = self.text_encoder(
            input_ids,
            attention_mask
        )

        fused_features, attention_weights = self.cross_attention(
            visual_features,
            text_features
        )

        sequence_input = fused_features.unsqueeze(1)

        sequence_output = self.sequence_model(sequence_input)

        final_output = sequence_output[:, -1, :]

        predictions = self.decoder(final_output)

        return predictions, attention_weights