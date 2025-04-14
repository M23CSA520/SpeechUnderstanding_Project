# ASR Fine-Tuning with Wav2Vec 2.0 for Hindi ( Low Resource Dataset )
# Overview
This project demonstrates fine-tuning the Wav2Vec 2.0 model for speech-to-text tasks in Hindi , a low-resource language. The model is fine-tuned using two publicly available datasets:

Mozilla Common Voice (Hindi Subset) : A crowdsourced dataset containing audio recordings and transcriptions in Hindi.
FLEURS (Few-shot LEarning Universal Representations of Speech) - Hindi : A high-quality multilingual dataset developed by Google.
The project highlights the challenges of working with low-resource languages and evaluates the model's performance using metrics like Word Error Rate (WER) and Character Error Rate (CER) .

# Table of Contents
Prerequisites
Installation
Dataset Description
Dataset Preparation
Fine-Tuning the Model
Evaluation
Results
Troubleshooting
Acknowledgments
Prerequisites
Before running the code, ensure the following:

Python >= 3.8
A GPU-enabled environment (e.g., Google Colab, local GPU, or cloud services like AWS/GCP).
Basic knowledge of PyTorch and Hugging Face's transformers library.
Installation
Install the required dependencies by running:

!pip install torch transformers datasets evaluate jiwer

# Dataset Description
Low-Resource Datasets
This project leverages two publicly available low-resource datasets for Hindi speech-to-text tasks:

Mozilla Common Voice (Hindi Subset) :
Description : Mozilla Common Voice is an open-source dataset that provides crowdsourced speech recordings and their transcriptions. The Hindi subset contains audio clips spoken by native and non-native Hindi speakers.
Features :
Speech recordings in .mp3 or .wav format.
Transcriptions in Hindi script.
Metadata including speaker demographics (e.g., accent, gender).
Challenges :
Noisy audio samples due to diverse recording environments.
Variability in pronunciation and accents.
Usage : Used for training and evaluating the fine-tuned Wav2Vec 2.0 model.
FLEURS (Few-shot LEarning Universal Representations of Speech) - Hindi :
Description : FLEURS is a multilingual speech dataset developed by Google. It includes high-quality speech recordings and transcriptions for over 100 languages, including Hindi.
Features :
Clean and high-quality audio recordings.
Balanced representation of male and female speakers.
Standardized transcriptions in Hindi script.
Challenges :
Limited size compared to high-resource languages like English.
Requires careful preprocessing to align with the model's input requirements.
Usage : Used as a supplementary dataset to improve the robustness of the fine-tuned model.
Dataset Preparation
To prepare the datasets for fine-tuning:

Download :
Download the Hindi subset of Common Voice and FLEURS from their respective sources:
Mozilla Common Voice
FLEURS Dataset
Preprocessing :
Convert audio files to a uniform format (e.g., .wav with a sampling rate of 16 kHz).
Normalize transcriptions to remove inconsistencies (e.g., extra spaces, punctuation).
Use the processor to extract audio features and tokenize text labels:

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch
Split :
Split the datasets into training, validation, and test sets using a ratio like 80%-10%-10%.
Fine-Tuning the Model
To fine-tune the Wav2Vec 2.0 model, follow these steps:

1. Load Pre-trained Model
Load the pre-trained Wav2Vec 2.0 model and processor:

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
2. Define Training Arguments
Configure the training arguments using TrainingArguments:

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=True,
    fp16=True,  # Enable mixed precision training
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
)
3. Create Trainer
Use the Trainer API for fine-tuning:

from transformers import Trainer, DataCollatorCTCWithPadding

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)
4. Start Training
Run the fine-tuning process:

trainer.train()
Evaluation
After fine-tuning, evaluate the model using WER and CER metrics:

# Results
Pre-trained Model Performance
Word Error Rate (WER) : 1.00
Character Error Rate (CER) : 1.76
Observations
The pre-trained model generates garbled and repetitive predictions, leading to poor performance on Hindi speech-to-text tasks.
Example:
Reference : वाराणसी हिंसा: पुलिस की भूमिका संदिग्ध, वायरल हुई तस्वीरें
Hypothesis : XFPXOFXZXOFXZXZXZXFOPOPUOPSPUXOXOEPZPOPXPRFOEOPEOPXOEROFXOXOF XOEPOFPUPEFOX KBPFOREXPXRXFPROXECROXEPOPXORFOEXEXZXOXZXZXOXZXPOX


# Acknowledgments
This project is built using the Hugging Face Transformers Library .
Special thanks to the authors of the Wav2Vec 2.0 paper: Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations .
Thanks to the contributors of the Mozilla Common Voice and FLEURS datasets for making them publicly available.





