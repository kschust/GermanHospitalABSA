#imports
import torch
import os

import spacy
import ast  # To safely evaluate strings as Python objects

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorWithPadding, pipeline, AutoModelForMaskedLM, BertForTokenClassification
from transformers import RobertaTokenizerFast, TrainerCallback
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import evaluate

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import shutil # delete model output dir

import time
import json
import torch.cuda as cuda
from GPUtil import getGPUs

class MetricsTracker:
    """
    Tracks and computes training performance metrics across epochs and batches.
    
    This utility class monitors:
    - Epoch timing statistics
    - Batch processing times
    - GPU memory usage (when available)
    - Hardware configuration
    
    Attributes:
        epoch_times (list): List of epoch durations in seconds
        batch_times (list): List of batch processing times in seconds
        memory_usage (list): Tracked GPU memory usage in MB (if available)
        current_gpu (object): Reference to GPU device if available
    
    Methods:
        start_epoch(): 
            Initializes timing for a new epoch and detects available GPU
        record_batch(): 
            Records completion of a single training batch
        end_epoch(): 
            Finalizes timing for the current epoch
        get_metrics(): 
            Computes and returns aggregated training metrics
    """
    def __init__(self):
        """Initialize metric tracking with empty containers."""
        self.epoch_times = []
        self.batch_times = []
        self.memory_usage = []
        self.current_gpu = None
        
    def start_epoch(self):
        """Start timing for a new epoch and detect available GPU resources."""
        self.epoch_start = time.time()
        self.batch_start = time.time()
        try:
            self.current_gpu = getGPUs()[0] if getGPUs() else None
        except:
            self.current_gpu = None
        
    def record_batch(self):
        """
        Record completion of a training batch.
        
        Records:
        - Batch processing time
        - Current GPU memory usage (if available)
        Resets batch timer for next measurement.
        """
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
        try:
            if self.current_gpu:
                self.memory_usage.append(self.current_gpu.memoryUsed)
        except:
            pass
        self.batch_start = time.time()
        
    def end_epoch(self):
        """Finalize timing for the current epoch."""
        self.epoch_times.append(time.time() - self.epoch_start)
        
    def get_metrics(self):
        """Compute and return aggregated training metrics."""
        return {
            "avg_epoch_time": sum(self.epoch_times)/len(self.epoch_times) if self.epoch_times else 0,
            "total_train_time": sum(self.epoch_times),
            "peak_memory": max(self.memory_usage) if self.memory_usage else 0,
            "avg_batch_time": sum(self.batch_times)/len(self.batch_times) if self.batch_times else 0,
            "gpu_type": self.current_gpu.name if self.current_gpu else "CPU"
        }

# Load the German spaCy model
nlp = spacy.load("de_core_news_sm")

# Ensure `aspectTerms` are parsed as dictionaries
def tokenize_and_annotate_ate(row):
    """
    Tokenizes text and generates aspect term annotations for ATE (Aspect Term Extraction) training.
    
    Args:
        row: DataFrame row containing:
            - raw_text: Original text string
            - aspectTerms: List of aspect term dictionaries (or string to be parsed)
    
    Returns:
        tuple: (tokens, labels) where:
            - tokens: List of word tokens
            - labels: List of corresponding BIO tags ('O' or 'B-ASPECT')

    Processing Logic:
    1. Parses aspectTerms (converts string to dict if needed)
    2. Tokenizes raw text using spaCy
    3. Initializes all labels as 'O' (non-aspect)
    4. Marks aspect terms with 'B-ASPECT' tags
    5. Returns aligned token-label pairs
    """
    # Parse the aspectTerms column into Python objects
    aspect_terms = row['aspectTerms']
    if isinstance(aspect_terms, str):  # Parse if it is a string
        aspect_terms = ast.literal_eval(aspect_terms)

    # Tokenize the text
    doc = nlp(row['raw_text'])
    tokens = [token.text for token in doc]
    labels = ['O'] * len(tokens)  # Initialize labels as 'O'

    # Annotate aspect terms with B tags # no I tags because to less exampes, models only predict B
    for aspect in aspect_terms:
        term, polarity = aspect['term'], aspect['polarity']
        if term != "noAspectTerm":  # Skip noAspectTerm
            term_tokens = nlp(term)
            start = -1
            for i in range(len(doc) - len(term_tokens) + 1):
                if [t.text for t in doc[i:i+len(term_tokens)]] == [t.text for t in term_tokens]:
                    start = i
                    break
            if start != -1:  # Mark B-I tags if the term is found
                labels[start] = 'B-ASPECT' 
                for j in range(1, len(term_tokens)):
                    labels[start + j] = 'B-ASPECT' 

    return tokens, labels

def tokenize_and_annotate_ate_cat(row):
    """
    Tokenizes text and generates aspect term annotations with category information.
    
    Args:
        row: DataFrame row containing:
            - raw_text: Input text
            - aspectTerms: List of aspect terms (str or list)
            - aspectCategories: Corresponding categories (str or list)
    
    Returns:
        tuple: (tokens, labels) where:
            - tokens: List of word tokens
            - labels: List of B-category tags (e.g., 'B-Arzt') or 'O'
    """
    aspect_terms = row['aspectTerms']
    aspect_categories = row['aspectCategories']
    
    if isinstance(aspect_terms, str):  
        aspect_terms = ast.literal_eval(aspect_terms)
    if isinstance(aspect_categories, str):  
        aspect_categories = ast.literal_eval(aspect_categories)
    
    # Tokenize the text
    doc = nlp(row['raw_text'])
    tokens = [token.text for token in doc]
    labels = ['O'] * len(tokens)  # Initialize labels as 'O'

    # Create a dictionary mapping aspect terms to their categories
    term_to_category = {aspect['term']: aspect_categories[i]['category'] 
                        for i, aspect in enumerate(aspect_terms) if aspect['term'] != "noAspectTerm"}

    # Annotate aspect terms with B B tags # no I tags because to less exampes, models only predict B
    for term, category in term_to_category.items():
        term_tokens = nlp(term)
        start = -1
        for i in range(len(doc) - len(term_tokens) + 1):
            if [t.text for t in doc[i:i+len(term_tokens)]] == [t.text for t in term_tokens]:
                start = i
                break
        if start != -1:  # Mark B-I tags if the term is found
            labels[start] = f'B-{category}'
            for j in range(1, len(term_tokens)):
                labels[start + j] = f'B-{category}'

    return tokens, labels


def tokenize_and_align_labels(tokenizer, examples):
    """
    Aligns tokenized inputs with word-level labels for sequence labeling.
    
    Args:
        tokenizer: Pretrained tokenizer
        examples: Batch containing:
            - text: Tokenized sentences
            - labels: Original word-level labels
    
    Returns:
        Dict: Tokenized inputs with:
            - input_ids: Token indices
            - attention_mask: Padding mask  
            - labels: Aligned labels (-100 for special tokens)
    """
    tokenized_inputs = tokenizer(
        examples["text"],
        padding=True,  # ensures uniform tensor sizes
        max_length=512,  # adjust based on your dataset
        #padding="max_length",
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label[word_idx] != "O" else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def prepare_data_for_ate_model(tokenizer, data):
    """
    Prepares ATE training data for model input.
    
    Args:
        tokenizer: Pretrained tokenizer  
        data: DataFrame with 'tokens_and_labels' tuples
    
    Returns:
        Dataset: HuggingFace Dataset with:
            - tokenized texts
            - aligned label IDs (0=O, 1=B-ASPECT)
            - attention masks
    
    Raises:
        ValueError: If label conversion fails
    """
    # Define mappings
    id2label = {0: "O", 1: "B-ASPECT"}
    label2id = {value: key for key, value in id2label.items()}
    # Extract texts and labels
    texts = [row['tokens_and_labels'][0] for _, row in data.iterrows()]
    labels = [row['tokens_and_labels'][1] for _, row in data.iterrows()]
    # Convert string labels to integer labels
    mapped_labels = []
    for label_list in labels:
        mapped_labels.append([label2id[label] for label in label_list])
    # Validate labels are now integers
    for i, label_list in enumerate(mapped_labels):
        if not all(isinstance(label, int) for label in label_list):
            print(f"Error in row {i}: {label_list}")
            raise ValueError("Mapped labels must be lists of integers.")
    # Create the dataset
    try:
        dataset = Dataset.from_dict({"text": texts, "labels": mapped_labels})
    except Exception as e:
        print("Failed to create dataset:")
        print("Texts:", texts)
        print("Labels:", mapped_labels)
        raise e
    return dataset.map(lambda examples: tokenize_and_align_labels(tokenizer, examples), batched=True)

def prepare_data_for_ate_cat_model(tokenizer, data):
    """
    Prepares category-aware ATE training data.
    
    Args:
        tokenizer: Pretrained tokenizer
        data: DataFrame with 'tokens_and_labels' tuples
    
    Returns:
        Dataset: HuggingFace Dataset with:
            - tokenized texts  
            - aligned label IDs (0=O, 1+=B-category)
            - attention masks
    """
    # Define unique categories from dataset
    categories = ["Allgemein", "anderer Service", "Arzt", "Krankenhaus", "mediz. Service", "Pflegepersonal", "Personal"]
    
    # Generate label mappings dynamically
    id2label = {0: "O"}
    label2id = {"O": 0}
    
    label_id = 1
    for cat in categories:
        id2label[label_id] = f"B-{cat}"
        label2id[f"B-{cat}"] = label_id
        label_id += 1

    # Extract texts and labels
    texts = [row['tokens_and_labels'][0] for _, row in data.iterrows()]
    labels = [row['tokens_and_labels'][1] for _, row in data.iterrows()]

    # Convert string labels to integer labels
    mapped_labels = []
    for label_list in labels:
        mapped_labels.append([label2id[label] for label in label_list])

    # Validate labels are now integers
    for i, label_list in enumerate(mapped_labels):
        if not all(isinstance(label, int) for label in label_list):
            print(f"Error in row {i}: {label_list}")
            raise ValueError("Mapped labels must be lists of integers.")

    # Create the dataset
    try:
        dataset = Dataset.from_dict({"text": texts, "labels": mapped_labels})
    except Exception as e:
        print("Failed to create dataset:")
        print("Texts:", texts)
        print("Labels:", mapped_labels)
        raise e

    return dataset.map(lambda examples: tokenize_and_align_labels(tokenizer, examples), batched=True)


# Define the function for evaluation metrics
def compute_metrics(p):
    """
    Computes evaluation metrics for basic Aspect Term Extraction (ATE).
    
    Args:
        p: Tuple of (predictions, labels) from model output
    
    Returns:
        Dict with precision, recall and F1 scores for:
        - Binary classification (O vs B-ASPECT)
        - Handles padding tokens (-100)
        - Provides fallback to 'O' for unknown predictions
    """
    predictions, labels = p
    # Flatten predictions and labels
    predictions = np.argmax(predictions, axis=-1)  # Take the argmax over the class dimension
    true_labels = []
    true_predictions = []
    # Define mappings
    id2label = {0: "O", 1: "B-ASPECT"}
    label2id = {value: key for key, value in id2label.items()}

    for label, pred in zip(labels, predictions):
        true_label = []
        true_pred = []
        for l, p in zip(label, pred):
            if l != -100:  # Ignore padding tokens
                true_label.append(id2label[l])  # Map ID back to label
                #true_pred.append(id2label[p])  # Map ID back to label

                # Debug: Check if predicted ID exists in id2label
                if p not in id2label:
                    print(f"Warning: Unexpected prediction {p} found. Assigning 'O'.")
                
                # Use .get() to avoid KeyError and assign "O" if missing
                true_pred.append(id2label.get(p, "O"))
                
        true_labels.append(true_label)
        true_predictions.append(true_pred)
    
    # Compute metrics using seqeval
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_metrics_cat(p):
    """
    Computes evaluation metrics for category-aware ATE.
    
    Args:
        p: Tuple of (predictions, labels) from model output
    
    Returns:
        Dict with precision, recall and F1 scores for:
        - Multi-class classification across 7 medical categories
        - Uses B-category tags (e.g., B-Arzt)
        - Robust to missing labels via fallback to 'O'
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    # Define full set of labels (adjust this list if needed)
    aspect_categories = ["Allgemein", "anderer Service", "Arzt", "Krankenhaus", "mediz. Service", "Pflegepersonal", "Personal"]
    base_labels = ["O"] + [f"B-{category}" for category in aspect_categories] # for prefix "B"]

    # Generate mappings
    id2label = {i: label for i, label in enumerate(base_labels)}
    label2id = {label: i for i, label in id2label.items()}

    true_labels, true_predictions = [], []

    for label_row, pred_row in zip(labels, predictions):
        true_label, true_pred = [], []
        for l, p in zip(label_row, pred_row):
            if l != -100:  # Ignore padding
                true_label.append(id2label.get(l, "O"))  # Fallback to "O" if key is missing

                # Debug: Check if predicted ID exists in id2label
                if p not in id2label:
                    print(f"Warning: Unexpected prediction {p} found. Assigning 'O'.")
                
                # Use .get() to avoid KeyError and assign "O" if missing
                true_pred.append(id2label.get(p, "O"))
            
        true_labels.append(true_label)
        true_predictions.append(true_pred)

    # Compute evaluation metrics
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    return {"precision": precision, "recall": recall, "f1": f1}

def print_ate_test_results(tokenizer, test_dataset, predictions, labels, output_file='testresult/ate_test_results.txt'):
    """
    Saves ATE model predictions with alignment to original text.
    
    Args:
        tokenizer: Model tokenizer for subword alignment  
        test_dataset: Evaluation dataset
        predictions: Raw model predictions
        labels: Ground truth labels
        output_file: Path to save results
    
    Outputs:
        - Formatted predictions showing token/label alignment
        - Console preview of first 2 samples
        - Handles subword tokenization properly
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    predicted_ids = np.argmax(predictions, axis=-1)
    id2label = {0: "O", 1: "B-ASPECT"}  

    with open(output_file, "w", encoding="utf-8") as f:  # Open file for writing
        for i in range(len(test_dataset)):
            sentence = test_dataset[i]["text"]  # Extract tokens from dataset
            tokenized_input = tokenizer(sentence, is_split_into_words=True, truncation=True, padding=True)
            word_ids = tokenized_input.word_ids()  # Map tokens to original words
    
            true_label_ids = labels[i]
            pred_label_ids = predicted_ids[i]
    
            aligned_true_labels = []
            aligned_pred_labels = []
            prev_word_idx = None
    
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None or true_label_ids[token_idx] == -100:
                    continue  # Skip special tokens and padding
    
                if word_idx != prev_word_idx:  # Only take first subword’s label
                    aligned_true_labels.append(id2label.get(true_label_ids[token_idx], "O"))
                    aligned_pred_labels.append(id2label.get(pred_label_ids[token_idx], "O"))
    
                prev_word_idx = word_idx

            # Write output to file
            f.write(f"Tokens     : {sentence}\n")
            f.write(f"True Labels: {aligned_true_labels}\n")
            f.write(f"Pred Labels: {aligned_pred_labels}\n")
            f.write("=" * 50 + "\n")

            if i < 2:
                print("Tokens     :", sentence)
                print("True Labels:", aligned_true_labels)
                print("Pred Labels:", aligned_pred_labels)
                print("=" * 50)

    print(f"Results saved to {output_file}") 

def print_ate_cat_test_results(tokenizer, test_dataset, predictions, labels, output_file='testresult/ate_cat_test_results.txt'):
    """
    Saves category-aware ATE predictions with label alignment.
    
    Args:
        tokenizer: Model tokenizer
        test_dataset: Evaluation data  
        predictions: Raw model outputs
        labels: Ground truth
        output_file: Save path
    
    Features:
    - Handles 7 medical categories
    - Aligns predictions with original tokens
    - Preserves B-category tags in output
    - Identical interface to base ATE version
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    predicted_ids = np.argmax(predictions, axis=-1)
    # Define full set of labels (adjust this list if needed)
    aspect_categories = ["Allgemein", "anderer Service", "Arzt", "Krankenhaus", "mediz. Service", "Pflegepersonal", "Personal"]
    base_labels = ["O"] + [f"B-{category}" for category in aspect_categories] # for prefix "B"

    # Generate mappings
    id2label = {i: label for i, label in enumerate(base_labels)}
    label2id = {label: i for i, label in id2label.items()}  

    with open(output_file, "w", encoding="utf-8") as f:  # Open file for writing
        for i in range(len(test_dataset)):
            sentence = test_dataset[i]["text"]  # Extract tokens from dataset
            tokenized_input = tokenizer(sentence, is_split_into_words=True, truncation=True, padding=True)
            word_ids = tokenized_input.word_ids()  # Map tokens to original words
    
            true_label_ids = labels[i]
            pred_label_ids = predicted_ids[i]
    
            aligned_true_labels = []
            aligned_pred_labels = []
            prev_word_idx = None
    
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None or true_label_ids[token_idx] == -100:
                    continue  # Skip special tokens and padding
    
                if word_idx != prev_word_idx:  # Only take first subword’s label
                    aligned_true_labels.append(id2label.get(true_label_ids[token_idx], "O"))
                    aligned_pred_labels.append(id2label.get(pred_label_ids[token_idx], "O"))
    
                prev_word_idx = word_idx

            # Write output to file
            f.write(f"Tokens     : {sentence}\n")
            f.write(f"True Labels: {aligned_true_labels}\n")
            f.write(f"Pred Labels: {aligned_pred_labels}\n")
            f.write("=" * 50 + "\n")

            if i < 2:
                print("Tokens     :", sentence)
                print("True Labels:", aligned_true_labels)
                print("Pred Labels:", aligned_pred_labels)
                print("=" * 50)

    print(f"Results saved to {output_file}")


def ate_model(data, model_name, rn1, rn2, epochs, save = False):
    """
    Trains and evaluates an Aspect Term Extraction (ATE) model with standard train-val-test split.
    
    Args:
        data: DataFrame containing text and aspect term annotations
        model_name: Pretrained model identifier (e.g., "bert-base-german")
        rn1: Random seed for initial train-test split
        rn2: Random seed for validation-test split  
        epochs: Number of training epochs
        save: Whether to save model checkpoints (default: False)
    
    Key Operations:
    1. Model Setup:
       - Handles different tokenizer types (BERT/RoBERTa)
       - Configures token classification head
    2. Data Processing:
       - 80-10-10 split (train-val-test)
       - Converts annotations to BIO tags
       - Computes class weights for imbalance
    3. Training:
       - Uses F1-optimized early stopping
       - Saves best model checkpoint
    4. Evaluation:
       - Generates classification report
       - Produces confusion matrix
       - Saves prediction examples
    
    Outputs:
    - Console metrics (precision/recall/F1)
    - Saved model artifacts (if save=True)
    - Test predictions and visualizations
    """
    if model_name in ["GerMedBERT/medbert-512", "TUM/GottBERT_base_last","TUM/GottBERT_base_best",
                      "TUM/GottBERT_filtered_base_best", "deepset/gbert-base"]:
        if model_name in ["GerMedBERT/medbert-512", "deepset/gbert-base"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        pipe = pipeline("fill-mask", model=model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)  # sequence labeling problem -> model predicts if token part of aspect term (B or O tags- (Begin, Outside))

    # Apply the function to your dataset
    data['tokens_and_labels'] = data.apply(tokenize_and_annotate_ate, axis=1)
    
    # First, split 80% for training and 20% for temporary (validation + test)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=rn1)
    # Then split the temporary set into 50% validation and 50% test (10% each of the original data)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=rn2)

    print("mapping of the data")
    print("")
    train_dataset = prepare_data_for_ate_model(tokenizer, train_data)
    val_dataset = prepare_data_for_ate_model(tokenizer, val_data)

    # class weights to handle imbalanced data -> isn't applied
    labels = [label for sublist in data['tokens_and_labels'].apply(lambda x: x[1]) for label in sublist]
    # Convert unique_labels to a NumPy array
    unique_labels = np.array(list(set(labels)))
    print(unique_labels)
    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
    # Convert to dictionary (mapping index to weight)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(class_weights_dict)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir= f"./models/BO/ate_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}",
        learning_rate=2e-5,  # standard
        per_device_train_batch_size=2, # size in which chunks are entered into the network, on how many data parallel weights are trained
        per_device_eval_batch_size= 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        weight_decay=0.01,
        metric_for_best_model="f1",  # Choose the best model based on F1-score
        greater_is_better=True,  # Since higher F1 is better
        load_best_model_at_end=True
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Padding -> map all tensors to the same size

    # Add metrics tracker
    metrics = MetricsTracker()
    # Custom callback for metrics
    class MetricsCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            metrics.start_epoch()
            
        def on_step_end(self, args, state, control, **kwargs):
            metrics.record_batch()
            
        def on_epoch_end(self, args, state, control, **kwargs):
            metrics.end_epoch()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback()]
    )

    print(f"Training results for {model_name} with {epochs} epochs and random seeds: {rn1}, {rn2}")
    print("")
    trainer.train()

    if save == True:
        trainer.save_model(f'./saved_models/ate_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        tokenizer.save_pretrained(f'./saved_tokenizers/ate_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        print(f'\nBest Model saved at: ./saved_models/ate_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        print(f'\nTokenizer for best Model saved at: ./saved_tokenizers/ate_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')

    print("mapping the test data")
    print("")
    test_dataset = prepare_data_for_ate_model(tokenizer, test_data)
    predictions, labels, _ = trainer.predict(test_dataset)

    # Convert predictions to class IDs
    predicted_ids = np.argmax(predictions, axis=-1)
    
    # Map predictions and labels back to their string labels
    true_labels = []
    true_predictions = []
    # Define mappings
    id2label = {0: "O", 1: "B-ASPECT"}
    label2id = {value: key for key, value in id2label.items()}

    # Check unique values before mapping
    unique_predicted_ids = set(np.unique(predicted_ids))
    unique_label_ids = set(id2label.keys())
    
    print("Unique predicted label IDs:", unique_predicted_ids)
    print("Expected label IDs:", unique_label_ids)
    
    for label_ids, pred_ids in zip(labels, predicted_ids):
        true_label = []
        true_pred = []
        for l, p in zip(label_ids, pred_ids):
            if l != -100:  # Ignore padding tokens
                true_label.append(id2label[l])

                # Debug: Check if predicted ID exists in id2label
                if p not in id2label:
                    print(f"Warning: Unexpected prediction {p} found. Assigning 'O_missing'.")
                
                # Use .get() to avoid KeyError and assign "O" if missing
                true_pred.append(id2label.get(p, "O_missing"))
                
        true_labels.append(true_label)
        true_predictions.append(true_pred)

    # Compute metrics
    print("Classification Report:")
    print(classification_report(true_labels, true_predictions))
    
    # Compute precision, recall, F1 score
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    # Call the function and save the results
    print_ate_test_results(tokenizer, test_dataset, predictions, labels, f'testresult/BO/ate/{epochs}_epochs/{model_name.replace("/", "_")}_ate_test_results.txt')

    # Flatten labels
    true_labels = [l for label_row in labels for l in label_row if l != -100]
    true_predictions = [p for pred_row, label_row in zip(predicted_ids, labels) for p, l in zip(pred_row, label_row) if l != -100]
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, true_predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["O", "B-ASPECT"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name} - {epochs} epochs")
    plt.xticks(rotation=90) # Rotate x-axis labels for readability
    plt.tight_layout() # Adjust layout to ensure no clipping of labels
    plt.savefig(f"testresult/BO/ate/{epochs}_epochs/{model_name.replace('/', '_')}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to testresult/ate/{epochs}_epochs/")

    # get performance metrics
    performance_metrics = metrics.get_metrics()
    print("\n=== Performance Metrics ===")
    print(f"GPU: {performance_metrics['gpu_type']}")
    print(f"Average epoch time: {performance_metrics['avg_epoch_time']:.2f}s")
    print(f"Total training time: {performance_metrics['total_train_time']:.2f}s")
    print(f"Peak GPU memory: {performance_metrics['peak_memory']}MB")
    print(f"Average batch time: {performance_metrics['avg_batch_time']:.4f}s")

    # Ensure directory exists
    os.makedirs(f"performanceresults/BO/ate/{epochs}_epochs", exist_ok=True)
    # Save to file
    with open(f"performanceresults/BO/ate/{epochs}_epochs/performance_{model_name.replace('/','_')}.txt", "w") as f:
        json.dump(performance_metrics, f)

    # Delete the saved model directory to free disk space
    output_dir= f"./models/BO/ate_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}"
    shutil.rmtree(output_dir, ignore_errors=True)
    print("Training complete. Model directory deleted to free memory.")


def ate_model_kfold(data, model_name, rn1, rn2, k, epochs):
    """
    Performs k-fold cross validation for ATE model training.
    
    Args:
        data: DataFrame with text and annotations  
        model_name: Pretrained model identifier
        rn1: Random seed for k-fold splitting
        rn2: Random seed for validation-test splitting
        k: Number of folds (default: 5)
        epochs: Training epochs per fold
    
    Key Features:
        - Stratified k-fold validation
        - Per-fold model training with:
          - Class-weighted loss
          - F1-optimized checkpointing
        - Aggregates metrics across folds
        - Cleans up temporary model files
    
    Returns:
        Prints average performance metrics across all folds:
        - Precision
        - Recall  
        - F1 score
    
    Key Differences from Standard Version:
        - Uses k-fold instead of single split
        - No model saving (focused on validation)
        - Reports cross-validated metrics
        - Handles larger datasets more robustly
    """
    if model_name in ["GerMedBERT/medbert-512", "TUM/GottBERT_base_last","TUM/GottBERT_base_best",
                      "TUM/GottBERT_filtered_base_best", "deepset/gbert-base"]:
        if model_name in ["GerMedBERT/medbert-512", "deepset/gbert-base"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        pipe = pipeline("fill-mask", model=model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)  # sequence labeling problem -> model predicts if token part of aspect term (B or O tags- (Begin, Outside))  

    data['tokens_and_labels'] = data.apply(tokenize_and_annotate_ate, axis=1)

    # K-Fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=rn1)
    fold_metrics = []
    all_precision, all_recall, all_f1 = [], [], []

    for fold, (train_idx, temp_idx) in enumerate(kf.split(data)):
        print(f"Starting fold {fold+1}/{k}")

        train_data = data.iloc[train_idx]
        temp_data = data.iloc[temp_idx]
        # Then split the temporary set into 50% validation and 50% test (10% each of the original data)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=rn2)

        train_dataset = prepare_data_for_ate_model(tokenizer, train_data)
        val_dataset = prepare_data_for_ate_model(tokenizer, val_data)
        test_dataset = prepare_data_for_ate_model(tokenizer, test_data)

        # class weights to handle imbalanced data
        labels = [label for sublist in data['tokens_and_labels'].apply(lambda x: x[1]) for label in sublist]
        # Convert unique_labels to a NumPy array
        unique_labels = np.array(list(set(labels)))
        print(unique_labels)
        # Compute class weights
        class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
        # Convert to dictionary (mapping index to weight)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(class_weights_dict)

        training_args = TrainingArguments(
            output_dir=f'./models/BO/ate_cv_k{k}_{model_name.replace("/", "_")}_fold{fold+1}',
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size= 2,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            weight_decay=0.01,
            metric_for_best_model="f1",  # Choose the best model based on F1-score
            greater_is_better=True,  # Since higher F1 is better
            load_best_model_at_end=True
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Add metrics tracker
        metrics = MetricsTracker()
        # Custom callback for metrics
        class MetricsCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                metrics.start_epoch()
                
            def on_step_end(self, args, state, control, **kwargs):
                metrics.record_batch()
                
            def on_epoch_end(self, args, state, control, **kwargs):
                metrics.end_epoch()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[MetricsCallback()]
        )

        print(f"Training fold {fold+1}")
        trainer.train()

        print(f"Evaluating fold {fold+1}")
        predictions, labels, _ = trainer.predict(test_dataset)
        # Store fold metrics
        fold_metrics.append(metrics.get_metrics())

        predicted_ids = np.argmax(predictions, axis=-1)
        true_labels, true_predictions = [], []

        id2label = {0: "O", 1: "B-ASPECT"}

        # Check unique values before mapping
        unique_predicted_ids = set(np.unique(predicted_ids))
        unique_label_ids = set(id2label.keys())
        
        print("Unique predicted label IDs:", unique_predicted_ids)
        print("Expected label IDs:", unique_label_ids)

        for label_ids, pred_ids in zip(labels, predicted_ids):
            true_label, true_pred = [], []
            for l, p in zip(label_ids, pred_ids):
                if l != -100:  
                    true_label.append(id2label[l])

                    # Debug: Check if predicted ID exists in id2label
                    if p not in id2label:
                        print(f"Warning: Unexpected prediction {p} found. Assigning 'O_missing'.")
                    
                    # Use .get() to avoid KeyError and assign "O" if missing
                    true_pred.append(id2label.get(p, "O_missing"))  
            true_labels.append(true_label)
            true_predictions.append(true_pred)

        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        print(f"Fold {fold+1} Results - Precision: {precision}, Recall: {recall}, F1: {f1}")

    print("\n=== Final Cross-Validation Results ===")
    print(f"Average Precision: {np.mean(all_precision)}")
    print(f"Average Recall: {np.mean(all_recall)}")
    print(f"Average F1 Score: {np.mean(all_f1)}")

    # Calculate average metrics from fold_metrics
    avg_metrics = {
        'avg_epoch_time': np.mean([m['avg_epoch_time'] for m in fold_metrics]),
        'total_train_time': np.sum([m['total_train_time'] for m in fold_metrics]),
        'peak_memory': max([m['peak_memory'] for m in fold_metrics]),
        'avg_batch_time': np.mean([m['avg_batch_time'] for m in fold_metrics]),
        'gpu_types': list(set([m['gpu_type'] for m in fold_metrics]))
    }

    # Print summary
    print(f"Average epoch time: {avg_metrics['avg_epoch_time']:.2f}s ± "
          f"{np.std([m['avg_epoch_time'] for m in fold_metrics]):.2f}s")
    print(f"Total training time: {avg_metrics['total_train_time']/60:.1f} minutes")
    print(f"Peak memory usage: {avg_metrics['peak_memory']}MB")
    print(f"Average batch time: {avg_metrics['avg_batch_time']:.4f}s ± "
          f"{np.std([m['avg_batch_time'] for m in fold_metrics]):.4f}s")
    print(f"GPUs used: {', '.join(avg_metrics['gpu_types'])}")
    
    # Save detailed metrics
    os.makedirs(f"performanceresults/BO/ate_{k}_fold/{epochs}_epochs", exist_ok=True)
    with open(f"performanceresults/BO/ate_{k}_fold/{epochs}_epochs/performance_{model_name.replace('/','_')}.json", "w") as f:
        json.dump({
            'per_fold': fold_metrics,
            'averages': avg_metrics,
            'task_metrics': {
                'precision': np.mean(all_precision),
                'recall': np.mean(all_recall),
                'f1': np.mean(all_f1)
            }
        }, f, indent=2)

    for i in range(1, k+1):
        # Delete the saved model directory to free disk space
        output_dir= f'./models/BO/ate_cv_k{k}_{model_name.replace("/", "_")}_fold{i}'
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Training complete. Model directory for fold {i} deleted to free memory.")

def ate_cat_model(data, model_name, rn1, rn2, epochs, save = False):
    """
    Trains and evaluates a category-aware Aspect Term Extraction (ATE) model.
    
    Args:
        data: DataFrame containing text, aspect terms and their categories
        model_name: Pretrained transformer model identifier
        rn1: Random seed for train-test split
        rn2: Random seed for validation-test split
        epochs: Number of training epochs
        save: Whether to save model checkpoints (default: False)
    
    Key Features:
        - Handles 7 medical categories (Arzt, Krankenhaus, etc.)
        - Uses B-category tags (e.g., B-Arzt) without I-tags
        - Automatic class weighting for imbalance
        - Saves:
          - Model predictions with category tags
          - Confusion matrix
          - Classification report
    
    Outputs:
        - Console metrics (precision/recall/F1 by category)
        - Saved model artifacts (if save=True)
        - Test predictions and visualizations
    """
    # Define unique categories
    categories = ["Allgemein", "anderer Service", "Arzt", "Krankenhaus", "mediz. Service", "Pflegepersonal", "Personal"]
    num_labels = 1 + len(categories)  # O + (B for each category)

    if model_name in ["GerMedBERT/medbert-512", "TUM/GottBERT_base_last","TUM/GottBERT_base_best",
                      "TUM/GottBERT_filtered_base_best", "deepset/gbert-base"]:
        if model_name in ["GerMedBERT/medbert-512", "deepset/gbert-base"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        pipe = pipeline("fill-mask", model=model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels) 

    # Apply tokenization and annotation
    data['tokens_and_labels'] = data.apply(tokenize_and_annotate_ate_cat, axis=1)

    # Split the data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=rn1)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=rn2)

    print("Mapping the data")
    train_dataset = prepare_data_for_ate_cat_model(tokenizer, train_data)
    val_dataset = prepare_data_for_ate_cat_model(tokenizer, val_data)

    # class weights to handle imbalanced data
    labels = [label for sublist in data['tokens_and_labels'].apply(lambda x: x[1]) for label in sublist]
    # Convert unique_labels to a NumPy array
    unique_labels = np.array(list(set(labels)))
    print(unique_labels)
    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
    # Convert to dictionary (mapping index to weight)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(class_weights_dict)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./models/BO/ate_cat_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}',
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size= 2,
        eval_strategy="epoch",
        save_strategy= "epoch",
        num_train_epochs=epochs,
        weight_decay=0.01,
        metric_for_best_model="f1",  # Choose the best model based on F1-score
        greater_is_better=True,  # Since higher F1 is better
        load_best_model_at_end=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Add metrics tracker
    metrics = MetricsTracker()
    # Custom callback for metrics
    class MetricsCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            metrics.start_epoch()
            
        def on_step_end(self, args, state, control, **kwargs):
            metrics.record_batch()
            
        def on_epoch_end(self, args, state, control, **kwargs):
            metrics.end_epoch()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_cat,
        callbacks=[MetricsCallback()]
    )

    print(f"Training {model_name} for {epochs} epochs with random seeds {rn1}, {rn2}")
    print("")
    trainer.train()

    if save == True:
        trainer.save_model(f'./saved_models/ate_cat_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        print(f'\nBest Model saved at: ./saved_models/ate_cat_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        tokenizer.save_pretrained(f'./saved_tokenizers/ate_cat_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        print(f'\nTokenizer for best Model saved at: ./saved_tokenizers/ate_cat_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        

    print("Evaluating on test data")
    print("")
    test_dataset = prepare_data_for_ate_cat_model(tokenizer, test_data)
    predictions, labels, _ = trainer.predict(test_dataset)

    # Convert predictions to class labels
    predicted_ids = np.argmax(predictions, axis=-1)

    # Map predictions and labels back to string labels
    id2label = {0: "O"}
    label_id = 1
    for cat in categories:
        id2label[label_id] = f"B-{cat}"
        label_id += 1
        """
        id2label[label_id] = f"I-{cat}"
        label_id += 1
        """

    # Check unique values before mapping
    unique_predicted_ids = set(np.unique(predicted_ids))
    unique_label_ids = set(id2label.keys())
    
    print("Unique predicted label IDs:", unique_predicted_ids)
    print("Expected label IDs:", unique_label_ids)
    
    true_labels = []
    true_predictions = []
    for label_ids, pred_ids in zip(labels, predicted_ids):
        true_label = []
        true_pred = []
        for l, p in zip(label_ids, pred_ids):
            if l != -100:  # Ignore padding tokens
                true_label.append(id2label[l])
    
                # Debug: Check if predicted ID exists in id2label
                if p not in id2label:
                    print(f"Warning: Unexpected prediction {p} found. Assigning 'O_missing'.")
                
                # Use .get() to avoid KeyError and assign "O" if missing
                true_pred.append(id2label.get(p, "O_missing"))
                
        true_labels.append(true_label)
        true_predictions.append(true_pred)

    print("Classification Report:")
    print(classification_report(true_labels, true_predictions))
    
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    # Call the function and save the results
    print_ate_cat_test_results(tokenizer, test_dataset, predictions, labels, f'testresult/BO/ate_cat/{epochs}_epochs/{model_name.replace("/", "_")}_ate_cat_test_results.txt')

    # Flatten labels
    true_labels = [l for label_row in labels for l in label_row if l != -100]
    true_predictions = [p for pred_row, label_row in zip(predicted_ids, labels) for p, l in zip(pred_row, label_row) if l != -100]
    # Confusion Matrix
    cm = confusion_matrix(true_labels, true_predictions, labels=list(id2label.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name} - {epochs} epochs")
    plt.xticks(rotation=90) # Rotate x-axis labels for readability
    plt.tight_layout() # Adjust layout to ensure no clipping of labels
    plt.savefig(f"testresult/BO/ate_cat/{epochs}_epochs/{model_name.replace('/', '_')}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to testresult/ate_cat/{epochs}_epochs/")

    # get performance metrics
    performance_metrics = metrics.get_metrics()
    print("\n=== Performance Metrics ===")
    print(f"GPU: {performance_metrics['gpu_type']}")
    print(f"Average epoch time: {performance_metrics['avg_epoch_time']:.2f}s")
    print(f"Total training time: {performance_metrics['total_train_time']:.2f}s")
    print(f"Peak GPU memory: {performance_metrics['peak_memory']}MB")
    print(f"Average batch time: {performance_metrics['avg_batch_time']:.4f}s")

    # Ensure directory exists
    os.makedirs(f"performanceresults/BO/ate_cat/{epochs}_epochs", exist_ok=True)
    # Save to file
    with open(f"performanceresults/BO/ate_cat/{epochs}_epochs/performance_{model_name.replace('/','_')}.txt", "w") as f:
        json.dump(performance_metrics, f)

    # Delete the saved model directory to free disk space
    output_dir= f'./models/BO/ate_cat_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}'
    shutil.rmtree(output_dir, ignore_errors=True)
    print("Training complete. Model directory deleted to free memory.")

def ate_cat_model_kfold(data, model_name, rn1, rn2, k, epochs):
    """
    Performs k-fold cross validation for category-aware ATE.
    
    Args:
        data: DataFrame with text and categorized aspects
        model_name: Pretrained model identifier
        rn1: Random seed for k-fold splitting
        rn2: Random seed for validation-test splitting  
        k: Number of folds (default: 5)
        epochs: Training epochs per fold
    
    Key Operations:
    1. Model Setup:
       - Dynamically generates label mappings for 7 categories
    2. Validation:
       - Per-fold metric tracking
       - Aggregates performance across folds
    3. Output:
       - Average precision/recall/F1
       - Cleanup of temporary model files
    """
    categories = ["Allgemein", "anderer Service", "Arzt", "Krankenhaus", "mediz. Service", "Pflegepersonal", "Personal"]
    num_labels = 1 + len(categories)   

    if model_name in ["GerMedBERT/medbert-512", "TUM/GottBERT_base_last","TUM/GottBERT_base_best",
                      "TUM/GottBERT_filtered_base_best", "deepset/gbert-base"]:
        if model_name in ["GerMedBERT/medbert-512", "deepset/gbert-base"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        pipe = pipeline("fill-mask", model=model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels) 

    data['tokens_and_labels'] = data.apply(tokenize_and_annotate_ate_cat, axis=1)

    # K-Fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=rn1)
    fold_metrics = []
    all_precision, all_recall, all_f1 = [], [], []

    for fold, (train_idx, temp_idx) in enumerate(kf.split(data)):
        print(f"Starting fold {fold+1}/{k}")

        train_data = data.iloc[train_idx]
        temp_data = data.iloc[temp_idx]
        # Then split the temporary set into 50% validation and 50% test (10% each of the original data)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=rn2)

        train_dataset = prepare_data_for_ate_cat_model(tokenizer, train_data)
        val_dataset = prepare_data_for_ate_cat_model(tokenizer, val_data)
        test_dataset = prepare_data_for_ate_cat_model(tokenizer, test_data)

        # class weights to handle imbalanced data
        labels = [label for sublist in data['tokens_and_labels'].apply(lambda x: x[1]) for label in sublist]
        # Convert unique_labels to a NumPy array
        unique_labels = np.array(list(set(labels)))
        print(unique_labels)
        # Compute class weights
        class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
        # Convert to dictionary (mapping index to weight)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(class_weights_dict)

        training_args = TrainingArguments(
            output_dir=f'./models/BO/ate_cat_cv_k{k}_{model_name.replace("/", "_")}_fold{fold+1}',
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            weight_decay=0.01,
             metric_for_best_model="f1",  # Choose the best model based on F1-score
            greater_is_better=True,  # Since higher F1 is better
            load_best_model_at_end=True
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Add metrics tracker
        metrics = MetricsTracker()
        # Custom callback for metrics
        class MetricsCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                metrics.start_epoch()
                
            def on_step_end(self, args, state, control, **kwargs):
                metrics.record_batch()
                
            def on_epoch_end(self, args, state, control, **kwargs):
                metrics.end_epoch()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_cat,
            callbacks=[MetricsCallback()]
        )

        print(f"Training fold {fold+1}")
        trainer.train()

        print(f"Evaluating fold {fold+1}")
        predictions, labels, _ = trainer.predict(test_dataset)
        # Store fold metrics
        fold_metrics.append(metrics.get_metrics())

        predicted_ids = np.argmax(predictions, axis=-1)
        id2label = {0: "O"}
        label_id = 1
        for cat in categories:
            id2label[label_id] = f"B-{cat}"
            label_id += 1
            """
            id2label[label_id] = f"I-{cat}"
            label_id += 1
            """
            
        # Check unique values before mapping
        unique_predicted_ids = set(np.unique(predicted_ids))
        unique_label_ids = set(id2label.keys())
        
        print("Unique predicted label IDs:", unique_predicted_ids)
        print("Expected label IDs:", unique_label_ids)
        
        true_labels = []
        true_predictions = []
        for label_ids, pred_ids in zip(labels, predicted_ids):
            true_label = []
            true_pred = []
            for l, p in zip(label_ids, pred_ids):
                if l != -100:  # Ignore padding tokens
                    true_label.append(id2label[l])
        
                    # Debug: Check if predicted ID exists in id2label
                    if p not in id2label:
                        print(f"Warning: Unexpected prediction {p} found. Assigning 'O_missing'.")
                    
                    # Use .get() to avoid KeyError and assign "O" if missing
                    true_pred.append(id2label.get(p, "O_missing"))
                    
            true_labels.append(true_label)
            true_predictions.append(true_pred)
        
        print("Classification Report:")
        print(classification_report(true_labels, true_predictions))
        
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        print(f"Fold {fold+1} Results - Precision: {precision}, Recall: {recall}, F1: {f1}")

    print("\n=== Final Cross-Validation Results ===")
    print(f"Average Precision: {np.mean(all_precision)}")
    print(f"Average Recall: {np.mean(all_recall)}")
    print(f"Average F1 Score: {np.mean(all_f1)}")

    # Calculate average metrics from fold_metrics
    avg_metrics = {
        'avg_epoch_time': np.mean([m['avg_epoch_time'] for m in fold_metrics]),
        'total_train_time': np.sum([m['total_train_time'] for m in fold_metrics]),
        'peak_memory': max([m['peak_memory'] for m in fold_metrics]),
        'avg_batch_time': np.mean([m['avg_batch_time'] for m in fold_metrics]),
        'gpu_types': list(set([m['gpu_type'] for m in fold_metrics]))
    }

    # Print summary
    print(f"Average epoch time: {avg_metrics['avg_epoch_time']:.2f}s ± "
          f"{np.std([m['avg_epoch_time'] for m in fold_metrics]):.2f}s")
    print(f"Total training time: {avg_metrics['total_train_time']/60:.1f} minutes")
    print(f"Peak memory usage: {avg_metrics['peak_memory']}MB")
    print(f"Average batch time: {avg_metrics['avg_batch_time']:.4f}s ± "
          f"{np.std([m['avg_batch_time'] for m in fold_metrics]):.4f}s")
    print(f"GPUs used: {', '.join(avg_metrics['gpu_types'])}")
    
    # Save detailed metrics
    os.makedirs(f"performanceresults/BO/ate_cat_{k}_fold/{epochs}_epochs", exist_ok=True)
    with open(f"performanceresults/BO/ate_cat_{k}_fold/{epochs}_epochs/performance_{model_name.replace('/','_')}.json", "w") as f:
        json.dump({
            'per_fold': fold_metrics,
            'averages': avg_metrics,
            'task_metrics': {
                'precision': np.mean(all_precision),
                'recall': np.mean(all_recall),
                'f1': np.mean(all_f1)
            }
        }, f, indent=2)

    for i in range(1, k+1):
        # Delete the saved model directory to free disk space
        output_dir= f'./models/BO/ate_cat_cv_k{k}_{model_name.replace("/", "_")}_fold{i}'
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Training complete. Model directory for fold {i} deleted to free memory.")
