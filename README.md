# Low-Resource Speech Recognition with Self-Supervised Learning

This project aims to improve Automatic Speech Recognition (ASR) in low-resource languages using self-supervised learning (SSL) techniques. We fine-tune pre-trained models like Wav2Vec 2.0 and XLS-R on publicly available datasets — Common Voice and FLEURS.

## 📚 Datasets Used

- **Common Voice** (Mozilla): Open multilingual dataset with community-contributed transcriptions.
- **FLEURS** (Google): Benchmark multilingual dataset for ASR tasks.

## 🎯 Objectives

- Improve ASR for underrepresented languages using minimal labeled data.
- Leverage pre-trained SSL models (Wav2Vec 2.0, XLS-R) via fine-tuning.
- Apply data augmentation for generalization.
- Evaluate using WER and CER metrics.

## 🧠 Methodology

1. **Preprocessing**  
   - Load and resample audio to 16kHz.  
   - Apply augmentation: time-stretch and pitch shift.

2. **Model**  
   - Use Hugging Face's `facebook/wav2vec2-base`.  
   - Train using Connectionist Temporal Classification (CTC) loss.  
   - Experiment with LoRA (Low-Rank Adaptation) for efficient fine-tuning.

3. **Evaluation**  
   - Metrics: Word Error Rate (WER) and Character Error Rate (CER).

## 🛠 How to Run

### 1. Install Requirements
```bash
pip install transformers datasets torch torchaudio soundfile jiwer librosa peft

### 2. Train on Common Voice or FLEURS
dset_name = 'C'  # 'C' for Common Voice, 'F' for FLEURS

Common Voice Dataset Statistics:
- Common Voice Train:
  - Number of samples: 4361
  - Total audio duration: 5.13 hours
- Common Voice Test:
  - Number of samples: 2894
  - Total audio duration: 3.98 hours
 
FLEURS Dataset Statistics:
- FLEURS Train:
  - Number of samples: 2120
  - Total audio duration: 6.66 hours
- FLEURS Test:
  - Number of samples: 418
  - Total audio duration: 1.34 hours

### 3. Preprocessing
Preprocessing :
Convert audio files to a uniform format (e.g., .wav with a sampling rate of 16 kHz).
Normalize transcriptions to remove inconsistencies (e.g., extra spaces, punctuation).
Use the processor to extract audio features and tokenize text labels.

### 4. Evaluate on Pretrained.

### 5. Fine tune model on CTC
     - a) Evaluate on Common Voice and Fleurs
### 6. Fine tune model on LoRA
     - a) Evaluate on Common Voice and Fleurs



Example:
Reference : वाराणसी हिंसा: पुलिस की भूमिका संदिग्ध, वायरल हुई तस्वीरें
Hypothesis : XFPXOFXZXOFXZXZXZXFOPOPUOPSPUXOXOEPZPOPXPRFOEOPEOPXOEROFXOXOF XOEPOFPUPEFOX KBPFOREXPXRXFPROXECROXEPOPXORFOEXEXZXOXZXZXOXZXPOX


## Acknowledgments
This project is built using the Hugging Face Transformers Library .
Special thanks to the authors of the Wav2Vec 2.0 paper: Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations .
Thanks to the contributors of the Mozilla Common Voice and FLEURS datasets for making them publicly available.

## 👨‍💻 Contributors
Niket Agrawal (M23CSA520) – m23csa520@iitj.ac.in
Ritesh Lamba (M23CSA544) – m23csa544@iitj.ac.in





