# -*- coding: utf-8 -*-
"""M23CSA520_M23CSA544_SU_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IYkqgZt5PCLvRv8KoT4N1K0a3S0McyBG
"""

# @title Install dependencies
!pip install transformers datasets torch torchaudio soundfile jiwer librosa

!pip install hf_xet
!pip install peft

# @title Import Necessary Librarires
import os
import random
import numpy as np
import torch
from datasets import load_dataset, load_from_disk, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from jiwer import wer, cer
import librosa
from sklearn.metrics import confusion_matrix
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# @title Device Set Up
if torch.cuda.is_available():
    print("CUDA is available! You can use GPU acceleration.")
else:
    print("CUDA is not available. You will be using the CPU.")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# @title Configuration Set up
# Configuration
class Config:
    # Dataset paths
    COMMON_VOICE_PATH = "mozilla-foundation/common_voice_11_0"
    FLEURS_PATH = "google/fleurs"

    # Pre-trained model
    MODEL_NAME = "facebook/wav2vec2-base"

    # Training parameters
    BATCH_SIZE = 2  # Reduced batch size for CPU
    NUM_EPOCHS = 5
    MAX_AUDIO_LEN = 16_000 * 10  # 10 seconds of audio
    SAMPLING_RATE = 16_000
    MAX_PREDICTION_LENGTH = 100
    WEIGHT_DECAY = 0.01

    # Output directories
    OUTPUT_DIR = "./results"
    OUTPUT_DIR1 = "./results_fleurs"

    LOG_DIR = "./logs"
    LOG_DIR1 = "./logs_fleurs"

# @title Data augmentation functions
def time_stretch(audio, rate=1.1):
    """Apply time stretching to the audio."""
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps=2):
    """Apply pitch shifting to the audio."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def augment_audio(audio, sr):
    """Apply random augmentations to the audio."""
    if random.random() > 0.5:  # 50% chance to apply time stretching
        audio = time_stretch(audio)
    if random.random() > 0.5:  # 50% chance to apply pitch shifting
        audio = pitch_shift(audio, sr)
    return audio

# @title Dataset Statistics
def print_dataset_statistics(dataset, name="Dataset"):
    """
    Prints the number of samples and total audio duration for a dataset.
    Args:
        dataset: Preprocessed dataset (train or test split).
        name (str): Name of the dataset (e.g., "Common Voice Train").
    """
    # Number of samples
    num_samples = len(dataset)

    # Total audio duration (in hours)
    total_duration = sum(len(batch["audio"]["array"]) / batch["audio"]["sampling_rate"] for batch in dataset) / 3600

    print(f"{name}:")
    print(f"  - Number of samples: {num_samples}")
    print(f"  - Total audio duration: {total_duration:.2f} hours")

# @title Preprocessing & Download Dataset function
def prepare_dataset(batch, processor, sr=16000, augment=False):
    """
    Preprocesses the dataset with optional data augmentation.
    Args:
        batch: A single example from the dataset.
        processor: Wav2Vec2Processor for feature extraction.
        sr: Sampling rate (default: 16kHz).
        augment (bool): Whether to apply data augmentation.
    Returns:
        dict: Processed batch with augmented or clean audio.
    """
    # Extract raw audio
    audio = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]

    # Resample audio if necessary
    if sampling_rate != sr:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)

    # Apply data augmentation (only if augment=True)
    if augment:
        audio = augment_audio(audio, sr)

    # Process audio
    inputs = processor(
        audio,
        sampling_rate=sr,
        max_length=Config.MAX_AUDIO_LEN,
        truncation=True
    )
    # Handle different column names for text transcription
    if "sentence" in batch:
        text_column = "sentence"
    elif "transcription" in batch:
        text_column = "transcription"
    else:
        raise KeyError("Dataset does not contain 'sentence' or 'transcription' column.")
    batch["input_values"] = torch.tensor(inputs.input_values[0]).to(device)
    batch["labels"] = torch.tensor(processor.tokenizer(batch[text_column]).input_ids).to(device)
    return batch

# Download and preprocess datasets
def download_and_preprocess_datasets(dset_name, language="hi", language_code="hi_in"):
    """
    Downloads and preprocesses Common Voice and FLEURS datasets.
    Args:
        dset_name (str): Dataset name (e.g., "C" for Common Voice, "F" for FLEURS).
        language (str): Language code for Common Voice (e.g., "hi").
        language_code (str): Language code for FLEURS (e.g., "hi_in").
    Returns:
        Tuple: Preprocessed train and test datasets.
    """
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_NAME)

    # Download and preprocess Common Voice
    if dset_name == 'C':
        asr_dset = load_dataset(Config.COMMON_VOICE_PATH, language, trust_remote_code=True)
    if dset_name == 'F':
        asr_dset = load_dataset(Config.FLEURS_PATH, language_code, trust_remote_code=True)

    asr_dset = asr_dset.cast_column("audio", Audio(sampling_rate=16000))
    if dset_name == 'C':
        print("Common Voice columns:", asr_dset["train"].column_names)
        print("Common Voice sample:", asr_dset["train"][0])

        # Print dataset statistics
        print("Common Voice Dataset Statistics:")
        print_dataset_statistics(asr_dset["train"], name="Common Voice Train")
        print_dataset_statistics(asr_dset["test"], name="Common Voice Test")
    if dset_name == 'F':
        print("FLEURS columns:", asr_dset["train"].column_names)
        print("FLEURS sample:", asr_dset["train"][0])

        # Print dataset statistics
        print("FLEURS Dataset Statistics:")
        print_dataset_statistics(asr_dset["train"], name="FLEURS Train")
        print_dataset_statistics(asr_dset["test"], name="FLEURS Test")
    if dset_name == 'C':
        asr_train = asr_dset["train"].map(
        lambda x: prepare_dataset(x, processor, augment=True),
        remove_columns=[col for col in asr_dset["train"].column_names if col not in ["audio", "sentence"]]
        )
        asr_test = asr_dset["test"].map(
        lambda x: prepare_dataset(x, processor, augment=False),
        remove_columns=[col for col in asr_dset["test"].column_names if col not in ["audio", "sentence"]]
        )
    if dset_name == 'F':
        asr_train = asr_dset["train"].map(
        lambda x: prepare_dataset(x, processor, augment=True),
        remove_columns=[col for col in asr_dset["train"].column_names if col not in ["audio", "transcription"]]
        )
        asr_test = asr_dset["test"].map(
        lambda x: prepare_dataset(x, processor, augment=False),
        remove_columns=[col for col in asr_dset["test"].column_names if col not in ["audio", "transcription"]]
        )

    return asr_train, asr_test, processor

# @title Evaluation on Pre Trained
from torch.cuda.amp import autocast

def evaluate_pretrained_model(model, processor, test_dataset):
    """
    Evaluates the pre-trained model on the test dataset.
    Args:
        model: Pre-trained Wav2Vec 2.0 model.
        processor: Wav2Vec2Processor for decoding predictions.
        test_dataset: Preprocessed test dataset.
    """
    references = []
    hypotheses = []

    # Determine the text column name dynamically
    if "sentence" in test_dataset.column_names:
        text_column = "sentence"
    elif "transcription" in test_dataset.column_names:
        text_column = "transcription"
    elif "text" in test_dataset.column_names:  # Add support for 'text' column
        text_column = "text"
    else:
        raise KeyError("Dataset does not contain 'sentence' or 'transcription' or 'text' column.")

    for batch in test_dataset:
        inputs = torch.tensor(batch["input_values"]).unsqueeze(0).to(device)
        with autocast():
            with torch.no_grad():
                logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(
            predicted_ids[0],
            skip_special_tokens=True)
        transcription = transcription[:Config.MAX_PREDICTION_LENGTH]
        # Remove repetitive patterns
        def remove_repetitions(text):
            return ''.join([text[i] for i in range(len(text)) if i == 0 or text[i] != text[i - 1]])

        transcription = remove_repetitions(transcription)
        references.append(batch[text_column])
        hypotheses.append(transcription)

    # Calculate Word Error Rate (WER)
    word_error_rate = wer(references, hypotheses)
    # Calculate Character Error Rate (CER)
    char_error_rate = cer(references, hypotheses)
    def normalize_text(text):
        return " ".join(text.strip().lower().split())

    references = [normalize_text(ref) for ref in references]
    hypotheses = [normalize_text(hyp) for hyp in hypotheses]
    print(f"Pre-trained Model Word Error Rate (WER): {word_error_rate:.2f}")
    print(f"Pre-trained Model Character Error Rate (CER): {char_error_rate:.2f}")
    for ref, hyp in zip(references[:5], hypotheses[:5]):
        print(f"Reference: {ref}")
        print(f"Hypothesis: {hyp}")
        print("-" * 50)

def tokenize(texts):
    """
    Tokenizes a list of texts into individual characters or words.
    Args:
        texts (list): List of strings to tokenize.
    Returns:
        list: Flattened list of tokens.
    """
    # Tokenize into characters
    return list("".join(texts))

!pip show transformers

# @title Custom Class for DataCollator with Padding
class CustomDataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # Split inputs and labels since they need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input values
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace labels with -100 where padding tokens are present
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Update batch with padded labels
        batch["labels"] = labels

        return batch



# @title Custom Class for Wav2Vec2ForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

class CustomWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def get_input_embeddings(self):
        import torch.nn as nn
        return nn.Identity()

# @title Function for Fine-tune the model
def fine_tune_model(train_dataset, test_dataset, processor):
    """
    Fine-tunes the Wav2Vec 2.0 Base model.
    Args:
        train_dataset: Preprocessed training dataset.
        test_dataset: Preprocessed test dataset.
        processor: Wav2Vec2Processor for feature extraction.
    """
    # Load pre-trained model
    model = Wav2Vec2ForCTC.from_pretrained(Config.MODEL_NAME)
    model.to(device)
    model.gradient_checkpointing_enable()

    NUM_TRAIN_EXAMPLES = len(train_dataset)

    # Calculate total steps
    total_steps = (NUM_TRAIN_EXAMPLES / Config.BATCH_SIZE) * Config.NUM_EPOCHS

    # Define warmup steps (10% of total steps)
    WARMUP_STEPS = int(0.1 * total_steps)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        eval_strategy="epoch",
        logging_dir=Config.LOG_DIR,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        fp16=True,
        learning_rate=5e-5,
        warmup_steps=WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        push_to_hub=False,
    )
    data_collator = CustomDataCollatorCTCWithPadding(processor=processor, padding=True)


    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,  # Use data collator for padding
    )

    # Start training
    trainer.train()
    trainer.save_model(Config.OUTPUT_DIR)

# @title Evaluation on Fine tuned Model
# Evaluate the model
def evaluate_model(model, processor, test_dataset):
    """
    Evaluates the fine-tuned model using Word Error Rate (WER).
    Args:
        model: Fine-tuned Wav2Vec 2.0 model.
        processor: Wav2Vec2Processor for decoding predictions.
        test_dataset: Preprocessed test dataset.
    """
    references = []
    hypotheses = []

        # Determine the text column name dynamically
    if "sentence" in test_dataset.column_names:
        text_column = "sentence"
    elif "transcription" in test_dataset.column_names:
        text_column = "transcription"
    else:
        raise KeyError("Dataset does not contain 'sentence' or 'transcription' column.")

    for batch in test_dataset:
        inputs = torch.tensor(batch["input_values"]).unsqueeze(0).to(device)
        with autocast():
            with torch.no_grad():
                logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(
            predicted_ids[0],
            skip_special_tokens=True)
        transcription = transcription[:Config.MAX_PREDICTION_LENGTH]
        # Remove repetitive patterns
        def remove_repetitions(text):
            return ''.join([text[i] for i in range(len(text)) if i == 0 or text[i] != text[i - 1]])

        transcription = remove_repetitions(transcription)
        references.append(batch[text_column])
        hypotheses.append(transcription)

    # Calculate Word Error Rate (WER)
    word_error_rate = wer(references, hypotheses)
    # Calculate Character Error Rate (CER)
    char_error_rate = cer(references, hypotheses)
    print(f"Fine Tuned Model - Word Error Rate (WER): {word_error_rate:.2f}")
    print(f"Fine Tuned Model - Character Error Rate (CER): {char_error_rate:.2f}")
    def normalize_text(text):
        return " ".join(text.strip().lower().split())

    references = [normalize_text(ref) for ref in references]
    hypotheses = [normalize_text(hyp) for hyp in hypotheses]
    for ref, hyp in zip(references[:5], hypotheses[:5]):
        print(f"Reference: {ref}")
        print(f"Hypothesis: {hyp}")
        print("-" * 50)

# @title Function for Fine Tuning
# Function for Fine-tune the model on fleurs with LoRA
from peft import LoraConfig, get_peft_model

def fine_tune_model_lora(train_dataset, test_dataset, processor):
    """
    Fine-tunes the Wav2Vec 2.0 Base model.
    Args:
        train_dataset: Preprocessed training dataset.
        test_dataset: Preprocessed test dataset.
        processor: Wav2Vec2Processor for feature extraction.
    """
    # Load pre-trained model
    lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.1,  # Dropout rate
    )
    model = CustomWav2Vec2ForCTC.from_pretrained(Config.MODEL_NAME)
    model = get_peft_model(model, lora_config)
    # Check if the model has input embeddings (for debugging purposes)

    model.to(device)
    model.gradient_checkpointing = False

    NUM_TRAIN_EXAMPLES = len(train_dataset)

    # Calculate total steps
    total_steps = (NUM_TRAIN_EXAMPLES / Config.BATCH_SIZE) * Config.NUM_EPOCHS

    # Define warmup steps (10% of total steps)
    WARMUP_STEPS = int(0.1 * total_steps)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        eval_strategy="epoch",
        logging_dir=Config.LOG_DIR,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        fp16=True,
        learning_rate=5e-5,
        warmup_steps=WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        push_to_hub=False,
    )
    data_collator = CustomDataCollatorCTCWithPadding(processor=processor, padding=True)
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,  # Use data collator for padding
    )

    # Start training
    trainer.train()
    model.config.adapter_attn_dim = Config.adapter_attn_dim
    trainer.save_model(Config.OUTPUT_DIR)

# @title Load Common Voice Dataset
# Step 1: Download and preprocess Common Voice Dataset
dset_name = 'C' #Common Voice
common_voice_train , common_voice_test , processor = download_and_preprocess_datasets(dset_name)

common_voice_train.save_to_disk("./common_voice_train")
common_voice_test.save_to_disk("./common_voice_test")

# @title Step 2: Load PreTrained model
pretrained_model = Wav2Vec2ForCTC.from_pretrained(Config.MODEL_NAME).to(device)
pretrained_model.gradient_checkpointing_enable()
print(f"Model is on device: {next(pretrained_model.parameters()).device}")

# @title Step 3: Evaluate pre-trained model on Common Voice datasets
print("Evaluating pre-trained model on Common Voice:")
evaluate_pretrained_model(pretrained_model, processor, common_voice_test)

# @title Step 4: Fine-tune and evaluate on Common Voice
print("Fine-tuning on Common Voice...")
Config.OUTPUT_DIR = "./fine_tuned_ctc_model_common_voice"
fine_tune_model(common_voice_train, common_voice_test, processor)
fine_tuned_model_ctc_cv = Wav2Vec2ForCTC.from_pretrained(Config.OUTPUT_DIR)
fine_tuned_model_ctc_cv.to(device)
print("Evaluating fine-tuned model on Common Voice:")
evaluate_model(fine_tuned_model_ctc_cv, processor, common_voice_test)

# @title Step 4: Fine-tune with LoRA and evaluate on Common Voice

print("Fine-tuning with LoRA on Common Voice...")
Config.adapter_attn_dim = 128
Config.OUTPUT_DIR = "./fine_tuned_lora_model_common_voice"
fine_tune_model_lora(common_voice_train, common_voice_test, processor)

# @title Evaluation on LoRA tuned Model
fine_tuned_model_lora_cv = CustomWav2Vec2ForCTC.from_pretrained(Config.OUTPUT_DIR)
fine_tuned_model_lora_cv.to(device)
print("Evaluating fine-tuned LoRA model on Common Voice:")
Config.adapter_attn_dim = 128
evaluate_model(fine_tuned_model_lora_cv, processor, common_voice_test)

# @title Step 5 : Clears RAM
# Save preprocessed Common Voice dataset to disk
common_voice_train.save_to_disk("./common_voice_train")
common_voice_test.save_to_disk("./common_voice_test")
del common_voice_train
del common_voice_test
gc.collect()

gc.collect()

# @title Loads Fleurs Dataset
#Step 1: Load Fleurs Datset
dset_name = 'F' #Fleurs Dataset
fleurs_train , fleurs_test , processor = download_and_preprocess_datasets(dset_name)

# @title Step 2: Load PreTrained model
pretrained_model = Wav2Vec2ForCTC.from_pretrained(Config.MODEL_NAME).to(device)
pretrained_model.gradient_checkpointing_enable()

# @title Step 3 : Evaluate pre-trained model on Fleurs datasets
print("Evaluating pre-trained model on FLEURS:")
evaluate_pretrained_model(pretrained_model, processor, fleurs_test)

# @title Fine tuning on Fleurs
print("Fine-tuning on Fleurs...")
Config.OUTPUT_DIR = "./fine_tuned_ctc_model_fleurs"
fine_tune_model(fleurs_train, fleurs_test, processor)
fine_tuned_model_ctc_fleurs = Wav2Vec2ForCTC.from_pretrained(Config.OUTPUT_DIR)
fine_tuned_model_ctc_fleurs.to(device)
print("Evaluating fine-tuned model on Fleurs :")
evaluate_model(fine_tuned_model_ctc_fleurs, processor, fleurs_test)

# @title Evaluation on Fine Tuned with Fleurs
Config.OUTPUT_DIR = "./fine_tuned_ctc_model_fleurs"
fine_tuned_model_ctc_fleurs = Wav2Vec2ForCTC.from_pretrained(Config.OUTPUT_DIR)
fine_tuned_model_ctc_fleurs.to(device)
print("Evaluating fine-tuned model on Fleurs :")
evaluate_model(fine_tuned_model_ctc_fleurs, processor, fleurs_test)

# @title Fine Tuning using LoRA on Fleurs Dataset
# Step 4: Fine-tune and evaluate on FLEURS
Config.adapter_attn_dim = 128
Config.OUTPUT_DIR = "./fine_tuned_lora_model_fleurs"
print("Fine-tuning on FLEURS...")
Config.adapter_attn_dim = 128
fine_tune_model_lora(fleurs_train, fleurs_test, processor)

# @title Evaluation on Fine Tuned LoRA
fine_tuned_model_lora_fleurs = CustomWav2Vec2ForCTC.from_pretrained(Config.OUTPUT_DIR, ignore_mismatched_sizes=True)
fine_tuned_model_lora_fleurs.config.adapter_attn_dim = Config.adapter_attn_dim  # or 128 directly
fine_tuned_model_lora_fleurs.to(device)
print("Evaluating fine-tuned model with LoRA on FLEURS:")
evaluate_model(fine_tuned_model_lora_fleurs, processor, fleurs_test)

# Save preprocessed Fleurs dataset to disk
fleurs_train.save_to_disk("./fleurs_train")
fleurs_test.save_to_disk("./fleurs_test")
del fleurs_train
del fleurs_test
gc.collect()
