import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import (
    load_storyreasoning_dataset,
    prepare_story_sample,
    tokenizer
)

from model import (
    VisualEncoder,
    TextEncoder
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


BASE_PATH = "/content/multimodal_story_reasoning"

RESULTS_PATH = f"{BASE_PATH}/results"

CHECKPOINT_PATH = f"{BASE_PATH}/checkpoints"

os.makedirs(RESULTS_PATH, exist_ok=True)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)


CONFIG = {

    "embed_dim": 256,

    "hidden_dim": 512,

    "vocab_size": tokenizer.vocab_size,

    "learning_rate": 1e-4,

    "batch_size": 4,

    "epochs": 2
}


def contrastive_loss(
    text_embeds,
    visual_embeds,
    temperature=0.07
):

    t = F.normalize(
        text_embeds,
        p=2,
        dim=1
    )

    v = F.normalize(
        visual_embeds,
        p=2,
        dim=1
    )

    similarity = torch.matmul(
        t,
        v.T
    ) / temperature

    labels = torch.arange(
        similarity.size(0),
        device=similarity.device
    )

    loss_t = F.cross_entropy(
        similarity,
        labels
    )

    loss_v = F.cross_entropy(
        similarity.T,
        labels
    )

    return (loss_t + loss_v) * 0.5


class StoryDataset(Dataset):

    def __init__(self, split):

        self.split = split

    def __len__(self):

        return len(self.split)

    def __getitem__(self, idx):

        sample = self.split[idx]

        processed = prepare_story_sample(sample)

        image_tensor = processed["images"][0]

        input_ids = processed["input_ids"]

        attention_mask = processed["attention_mask"]

        return {

            "image": image_tensor,

            "input_ids": input_ids,

            "attention_mask": attention_mask,

            "target_ids": input_ids
        }


class StoryReasoningModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.visual_enc = VisualEncoder(
            config["embed_dim"]
        )

        self.text_enc = TextEncoder(
            config["embed_dim"]
        )

        self.fusion_fc = nn.Linear(
            config["embed_dim"] * 2,
            config["hidden_dim"]
        )

        self.temporal_lstm = nn.LSTM(
            input_size=config["hidden_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=1,
            batch_first=True
        )

        self.text_embed = nn.Embedding(
            config["vocab_size"],
            config["embed_dim"]
        )

        self.text_decoder_gru = nn.GRU(
            config["embed_dim"],
            config["hidden_dim"],
            batch_first=True
        )

        self.text_out_fc = nn.Linear(
            config["hidden_dim"],
            config["vocab_size"]
        )

        self.image_decoder = nn.Sequential(

            nn.Linear(
                config["hidden_dim"],
                256 * 7 * 7
            ),

            nn.ReLU(),

            nn.Unflatten(
                dim=1,
                unflattened_size=(256, 7, 7)
            ),

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(32,16,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(16,3,4,2,1),

            nn.Sigmoid()
        )

    def forward(
        self,
        images,
        input_ids,
        mask,
        target_ids=None
    ):

        v_emb = self.visual_enc(images)

        t_emb = self.text_enc(
            input_ids,
            mask
        )

        if len(v_emb.shape) == 2:

            v_emb = v_emb.unsqueeze(1)

        if len(t_emb.shape) == 2:

            t_emb = t_emb.unsqueeze(1)

        B, S, E = v_emb.size()

        align_loss = contrastive_loss(

            t_emb.reshape(B*S, E),

            v_emb.reshape(B*S, E)
        )

        fused = torch.cat(
            (v_emb, t_emb),
            dim=-1
        )

        fused = F.relu(
            self.fusion_fc(fused)
        )

        _, (h_last, _) = self.temporal_lstm(fused)

        context_vector = h_last[-1]

        pred_img = self.image_decoder(
            context_vector
        )

        text_logits = None

        if target_ids is not None:

            tgt_embed = self.text_embed(
                target_ids
            )

            decoder_init = context_vector.unsqueeze(0)

            dec_out, _ = self.text_decoder_gru(
                tgt_embed,
                decoder_init
            )

            text_logits = self.text_out_fc(
                dec_out
            )

        return pred_img, text_logits, align_loss


dataset = load_storyreasoning_dataset()

train_split = dataset["train"].select(range(500))

test_split = dataset["test"].select(range(100))


train_dataset = StoryDataset(train_split)

test_dataset = StoryDataset(test_split)


train_loader = DataLoader(

    train_dataset,

    batch_size=CONFIG["batch_size"],

    shuffle=True
)


test_loader = DataLoader(

    test_dataset,

    batch_size=CONFIG["batch_size"],

    shuffle=False
)


model = StoryReasoningModel(CONFIG).to(device)

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=CONFIG["learning_rate"]
)


criterion = nn.CrossEntropyLoss()


architecture_file = (
    f"{RESULTS_PATH}/model_outputs/"
    "multimodal_model_architecture.txt"
)

os.makedirs(
    f"{RESULTS_PATH}/model_outputs",
    exist_ok=True
)


with open(architecture_file, "w") as f:

    f.write(
        "MULTIMODAL STORY REASONING MODEL\n\n"
    )

    f.write(str(model))

    f.write("\n\nCONFIGURATION\n\n")

    for k, v in CONFIG.items():

        f.write(f"{k}: {v}\n")


print("Model architecture saved.\n")


for epoch in range(CONFIG["epochs"]):

    model.train()

    total_loss = 0

    progress = tqdm(train_loader)

    for batch in progress:

        imgs = batch["image"].to(device)

        inp_ids = batch["input_ids"].to(device)

        mask = batch["attention_mask"].to(device)

        tgt_ids = batch["target_ids"].to(device)

        optimizer.zero_grad()

        pred_img, text_logits, align_loss = model(

            imgs,

            inp_ids,

            mask,

            tgt_ids
        )

        text_loss = criterion(

            text_logits.reshape(
                -1,
                CONFIG["vocab_size"]
            ),

            tgt_ids.reshape(-1)
        )

        loss = text_loss + align_loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1} Loss {loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)

    print(
        f"\nEpoch {epoch+1} Average Loss: "
        f"{avg_loss:.4f}"
    )


checkpoint_file = (
    f"{CHECKPOINT_PATH}/"
    "story_reasoning_model.pth"
)

torch.save(
    model.state_dict(),
    checkpoint_file
)

print("\nTraining completed successfully.")

print(f"\nCheckpoint saved at:\n{checkpoint_file}")