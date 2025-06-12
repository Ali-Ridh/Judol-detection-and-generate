import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset as HFDataset
import gradio as gr
import numpy as np
import sys
import os
from tqdm.auto import tqdm

# --- General Configuration ---
TEXT_COLUMN = "cleaned_message"
LABEL_COLUMN = "label"
CSV_FILE_PATH = "youtube_chat_jogja_clean.csv"
RANDOM_STATE = 42

# --- Detector (BERT + BiLSTM) Configuration ---
DETECTOR_BASE_MODEL = "cahya/bert-base-indonesian-522M"
OUTPUT_DIR_DETECTOR = "./judol_ad_detector_bert_bilstm_model"
DETECTOR_MAX_LENGTH = 128
DETECTOR_BATCH_SIZE = 16
DETECTOR_LEARNING_RATE = 2e-5
LSTM_HIDDEN_DIM = 256
LSTM_N_LAYERS = 2
LSTM_DROPOUT = 0.25

# --- Generator (GPT-2) Configuration ---
GENERATOR_BASE_MODEL = "cahya/gpt2-small-indonesian-522M"
OUTPUT_DIR_GENERATOR = "./judol_ad_generator_model"
GENERATOR_MAX_LENGTH = 128
GENERATOR_BATCH_SIZE = 8
GENERATOR_LEARNING_RATE = 2e-5
TRAINING_EPOCHS = 3 # Used for both models

# --- Global variables ---
g_detector_model, g_detector_tokenizer = None, None
g_generator_model, g_generator_tokenizer = None, None
g_device = None

#===============================================================
#  1. DETECTOR MODEL (BERT + BiLSTM) DEFINITION AND FUNCTIONS
#===============================================================
class BERT_BiLSTM_Classifier(nn.Module):
    def __init__(self, model_name, hidden_dim, n_layers, dropout):
        super(BERT_BiLSTM_Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size, hidden_size=hidden_dim,
            num_layers=n_layers, bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_dim * 2, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        _, (hidden, _) = self.lstm(sequence_output)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}

class DetectorDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]), add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }

def train_detector_model():
    """Trains and saves the detector model."""
    global g_detector_model, g_detector_tokenizer, g_device
    print("--- Starting Detector Model Training ---")
    
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    except Exception as e:
        print(f"Error loading detector data: {e}")
        return False

    train_df, _ = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN])
    
    g_detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_BASE_MODEL)
    train_dataset = DetectorDataset(
        train_df[TEXT_COLUMN].tolist(), train_df[LABEL_COLUMN].tolist(), 
        g_detector_tokenizer, DETECTOR_MAX_LENGTH
    )
    train_dataloader = DataLoader(train_dataset, batch_size=DETECTOR_BATCH_SIZE, shuffle=True)

    g_detector_model = BERT_BiLSTM_Classifier(DETECTOR_BASE_MODEL, LSTM_HIDDEN_DIM, LSTM_N_LAYERS, LSTM_DROPOUT)
    g_detector_model.to(g_device)
    
    optimizer = AdamW(g_detector_model.parameters(), lr=DETECTOR_LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_dataloader) * TRAINING_EPOCHS)

    for epoch in range(TRAINING_EPOCHS):
        print(f"\nDetector Training - Epoch {epoch + 1}/{TRAINING_EPOCHS}")
        g_detector_model.train()
        progress_bar = tqdm(train_dataloader, desc="Training Detector")
        for batch in progress_bar:
            optimizer.zero_grad()
            outputs = g_detector_model(
                input_ids=batch['input_ids'].to(g_device),
                attention_mask=batch['attention_mask'].to(g_device),
                labels=batch['labels'].to(g_device)
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'loss': loss.item()})
    
    if not os.path.exists(OUTPUT_DIR_DETECTOR): os.makedirs(OUTPUT_DIR_DETECTOR)
    torch.save(g_detector_model, os.path.join(OUTPUT_DIR_DETECTOR, "bert_bilstm_model.pt"))
    g_detector_tokenizer.save_pretrained(OUTPUT_DIR_DETECTOR)
    print("Detector model training complete and saved.")
    g_detector_model.eval()
    return True

def predict_detection(text):
    """Prediction function for the detector GUI."""
    global g_detector_model, g_detector_tokenizer, g_device
    if not all([g_detector_model, g_detector_tokenizer]): return "Detector model not loaded."

    inputs = g_detector_tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=DETECTOR_MAX_LENGTH,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        outputs = g_detector_model(
            input_ids=inputs['input_ids'].to(g_device),
            attention_mask=inputs['attention_mask'].to(g_device)
        )
    
    probabilities = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()[0]
    judol_ad_prob = probabilities[1]
    
    category = "Tinggi (High)" if judol_ad_prob > 0.75 else "Menengah (Medium)" if judol_ad_prob > 0.40 else "Rendah (Low)"
    return f"### Hasil Deteksi\n---\n**Tingkat Deteksi:** `{category}`\n\n**Probabilitas Iklan Judol:** `{judol_ad_prob*100:.2f}%`"

#===============================================================
#      2. GENERATOR MODEL (GPT-2) DEFINITION AND FUNCTIONS
#===============================================================
def fine_tune_generator_model():
    """Fine-tunes and saves the generator model."""
    global g_generator_model, g_generator_tokenizer, g_device
    print("\n--- Starting Generator Model Fine-Tuning ---")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        judol_ads_df = df[df[LABEL_COLUMN] == 1].copy()
        if len(judol_ads_df) == 0:
            print("No judol ad data (label 1) found for generator fine-tuning.")
            return False
    except Exception as e:
        print(f"Error loading generator data: {e}")
        return False
        
    hg_dataset = HFDataset.from_pandas(judol_ads_df)
    
    g_generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_BASE_MODEL)
    g_generator_model = AutoModelForCausalLM.from_pretrained(GENERATOR_BASE_MODEL)
    g_generator_tokenizer.pad_token = g_generator_tokenizer.eos_token
    g_generator_model.config.pad_token_id = g_generator_model.config.eos_token_id

    def tokenize_function(examples):
        tokenized = g_generator_tokenizer(
            [txt + g_generator_tokenizer.eos_token for txt in examples[TEXT_COLUMN]],
            truncation=True, max_length=GENERATOR_MAX_LENGTH, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = hg_dataset.map(tokenize_function, batched=True, remove_columns=hg_dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_GENERATOR, overwrite_output_dir=True,
        num_train_epochs=TRAINING_EPOCHS, per_device_train_batch_size=GENERATOR_BATCH_SIZE,
        save_total_limit=1, prediction_loss_only=True, logging_steps=100
    )

    trainer = Trainer(
        model=g_generator_model, args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=g_generator_tokenizer, mlm=False),
        train_dataset=tokenized_dataset
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR_GENERATOR)
    g_generator_tokenizer.save_pretrained(OUTPUT_DIR_GENERATOR)
    print("Generator model fine-tuning complete and saved.")
    g_generator_model.eval()
    return True

def generate_ad(prompt_text, max_new_tokens=50):
    """Generation function for the generator GUI."""
    global g_generator_model, g_generator_tokenizer, g_device
    if not all([g_generator_model, g_generator_tokenizer]): return "Generator model not loaded."

    inputs = g_generator_tokenizer(prompt_text, return_tensors="pt").to(g_device)
    with torch.no_grad():
        outputs = g_generator_model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_return_sequences=1,
            do_sample=True, top_k=50, top_p=0.95, temperature=0.8,
            pad_token_id=g_generator_tokenizer.eos_token_id
        )
    return g_generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

#===============================================================
#                         3. MAIN ORCHESTRATOR
#===============================================================
def main():
    """Main function to train/load models and launch the combined GUI."""
    global g_device
    global g_detector_model, g_detector_tokenizer
    global g_generator_model, g_generator_tokenizer
    
    g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main device set to: {g_device}")

    # --- Setup Detector Model ---
    detector_model_path = os.path.join(OUTPUT_DIR_DETECTOR, "bert_bilstm_model.pt")
    if os.path.exists(detector_model_path):
        print(f"Found existing detector model. Loading...")
        # FIX: Added weights_only=False to allow loading of custom model class
        g_detector_model = torch.load(detector_model_path, map_location=g_device, weights_only=False)
        g_detector_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR_DETECTOR)
        g_detector_model.to(g_device)
        g_detector_model.eval()
    else:
        if not train_detector_model():
            print("Detector training failed. The detection tab may not work.")
    
    # --- Setup Generator Model ---
    generator_model_path = os.path.join(OUTPUT_DIR_GENERATOR, "pytorch_model.bin")
    if os.path.exists(generator_model_path):
        print(f"Found existing generator model. Loading...")
        g_generator_model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR_GENERATOR)
        g_generator_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR_GENERATOR)
        g_generator_model.to(g_device)
        g_generator_model.eval()
    else:
        if not fine_tune_generator_model():
            print("Generator fine-tuning failed. The generation tab may not work.")

    # --- Create and Launch Tabbed GUI ---
    print("\n--- Launching Combined Gradio Interface ---")
    
    detector_iface = gr.Interface(
        fn=predict_detection,
        inputs=gr.Textbox(lines=3, placeholder="Enter text to check for judol ads..."),
        outputs=gr.Markdown(),
        title="Detector Iklan Judol (BERT + BiLSTM)",
        description="Check if a piece of text is likely a judol ad.",
        examples=[["ayo main slot gacor hari ini, wd cepat!"], ["besok ada acara apa ya di jogja?"]]
    )
    
    generator_iface = gr.Interface(
        fn=generate_ad,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter a starting phrase..."),
            gr.Slider(10, 100, step=5, value=50, label="Max New Tokens")
        ],
        outputs=gr.Textbox(label="Generated Judol Ad"),
        title="Generator Iklan Judol",
        description="Generate judol ad-style text from a prompt. For educational purposes only.",
        examples=[["situs slot terbaik", 50], ["bonus member baru", 60]]
    )
    
    tabbed_interface = gr.TabbedInterface(
        [detector_iface, generator_iface],
        ["Deteksi Iklan (Ad Detection)", "Generator Iklan (Ad Generation)"]
    )
    
    tabbed_interface.launch(share=True)

if __name__ == "__main__":
    main()