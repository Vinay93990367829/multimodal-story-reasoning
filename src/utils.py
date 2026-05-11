import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer
from torchvision import transforms
from nltk.translate.meteor_score import meteor_score


nltk.download("wordnet")
nltk.download("omw-1.4")


RESULTS_DIR = "multimodal_story_reasoning/results"


bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


image_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def save_plot(filename):

    save_path = os.path.join(
        RESULTS_DIR,
        "plots",
        filename
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    print(f"Plot saved to: {save_path}")


def save_table(dataframe, filename):

    save_path = os.path.join(
        RESULTS_DIR,
        "tables",
        filename
    )

    dataframe.to_csv(
        save_path,
        index=False
    )

    print(f"Table saved to: {save_path}")


def save_metrics(metrics_dict, filename):

    df = pd.DataFrame([metrics_dict])

    save_path = os.path.join(
        RESULTS_DIR,
        "metrics",
        filename
    )

    df.to_csv(
        save_path,
        index=False
    )

    print(f"Metrics saved to: {save_path}")


def save_figure(filename):

    save_path = os.path.join(
        RESULTS_DIR,
        "figures",
        filename
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    print(f"Figure saved to: {save_path}")


def save_prediction_text(predictions, filename):

    save_path = os.path.join(
        RESULTS_DIR,
        "predictions",
        filename
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in predictions:

            f.write(str(item) + "\n")

    print(f"Predictions saved to: {save_path}")


def save_model_output(output, filename):

    save_path = os.path.join(
        RESULTS_DIR,
        "model_outputs",
        filename
    )

    torch.save(output, save_path)

    print(f"Model output saved to: {save_path}")


def load_storyreasoning_dataset():

    dataset = load_dataset(
        "daniel3303/StoryReasoning"
    )

    return dataset


def tokenize_text(text, max_length=64):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return encoding


def process_image(image):

    image = image.convert("RGB")

    image = image_transform(image)

    return image


def prepare_story_sample(sample):

    images = sample["images"]

    story_text = sample["story"]

    processed_images = []

    for image in images:

        processed_image = process_image(image)

        processed_images.append(processed_image)

    text_encoding = tokenize_text(story_text)

    return {

        "images": processed_images,

        "input_ids":
            text_encoding["input_ids"].squeeze(0),

        "attention_mask":
            text_encoding["attention_mask"].squeeze(0)
    }


def display_story_sequence(images):

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(20, 5)
    )

    for i, image in enumerate(images):

        if torch.is_tensor(image):

            image = image.permute(1, 2, 0).numpy()

        axes[i].imshow(image)

        axes[i].axis("off")

        axes[i].set_title(f"Frame {i+1}")

    plt.tight_layout()


def calculate_bleu(predictions, references):

    formatted_references = [
        [ref] for ref in references
    ]

    results = bleu_metric.compute(
        predictions=predictions,
        references=formatted_references
    )

    return results["bleu"]


def calculate_rouge(predictions, references):

    results = rouge_metric.compute(
        predictions=predictions,
        references=references
    )

    return results["rougeL"]


def calculate_meteor(predictions, references):

    meteor_scores = []

    for pred, ref in zip(
        predictions,
        references
    ):

        score = meteor_score(
            [ref.split()],
            pred.split()
        )

        meteor_scores.append(score)

    average_score = (
        sum(meteor_scores)
        / len(meteor_scores)
    )

    return average_score