# imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import EvalPrediction, DataCollatorWithPadding
from transformers import RobertaTokenizerFast, AutoModelForMaskedLM, pipeline, AutoModelForTokenClassification
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
from datasets import Dataset
from functools import partial

#import ast
import json
import os
import shutil # delete model output dir

from sklearn.model_selection import train_test_split
from seqeval.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

from transformers import EarlyStoppingCallback
from transformers import TrainerCallback

import time
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

class RelativeEarlyStopping(TrainerCallback):
    """
    Custom early stopping callback that monitors relative F1 improvement.
    
    Args:
        patience (int): Number of epochs to wait before stopping if no improvement (default: 3)
        min_improvement (float): Minimum relative F1 improvement required (default: 0.001)
        min_epochs (int): Minimum training epochs before checking for early stopping (default: 5)
    
    Behavior:
    - Only activates after min_epochs have completed
    - Tracks F1 score improvements between epochs
    - Stops training if improvement < min_improvement for patience epochs
    - Resets counter when significant improvement occurs
    """
    def __init__(self, patience=3, min_improvement=0.001, min_epochs=5):
        """Initialize early stopping parameters"""
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_epochs = min_epochs
        self.last_f1 = None
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Callback that checks for improvement after each evaluation"""
        current_f1 = metrics.get("eval_f1", None)
        if current_f1 is None or state.epoch < self.min_epochs:
            return

        if self.last_f1 is None:
            self.last_f1 = current_f1
            return

        # Continue if F1 improves over previous epoch by at least min_improvement
        if current_f1 >= (self.last_f1 + self.min_improvement):
            self.no_improvement_count = 0  # Reset counter
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                control.should_training_stop = True  # Stop training

        self.last_f1 = current_f1  # Update for next epoch

def tokenize_function(tokenizer, examples):
    """
    Standard text tokenization function for HuggingFace models.
    
    Args:
        tokenizer: Pretrained tokenizer instance
        examples: Batch of text samples to tokenize
    
    Returns:
        Dictionary containing:
        - input_ids: Tokenized text
        - attention_mask: Padding mask
        - Other tokenizer outputs
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512) 

def compute_metrics(eval_pred):
    """
    Calculates classification metrics during model evaluation.
    
    Args:
        eval_pred: Tuple of (logits, true_labels)
    
    Returns:
        Dictionary containing:
        - accuracy: Overall correct predictions
        - precision: Weighted by class support
        - recall: Weighted by class support  
        - f1: Weighted harmonic mean
        - class_distribution: Prediction counts per class

    Notes:
        - Uses weighted averaging for imbalanced datasets
        - Handles zero-division cases safely
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "class_distribution": {i: np.sum(predictions == i) for i in range(3)}}

def compute_metrics_kfold(eval_pred):
    """
    Computes evaluation metrics for k-fold cross validation.
    
    Args:
        eval_pred: Tuple containing (logits, true_labels) from model predictions
    
    Returns:
        Dictionary with:
        - eval_accuracy: Overall accuracy score (float)
        - eval_precision: Weighted precision (float)
        - eval_recall: Weighted recall (float) 
        - eval_f1: Weighted F1-score (float)
        - class_distribution: Count of predictions per class (dict)

    Note: Uses weighted averaging for imbalanced classes and converts all metrics to floats.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    # Return only numerical metrics, not dictionaries
    return {
        "eval_accuracy": float(accuracy), "eval_precision": float(precision), "eval_recall": float(recall), "eval_f1": float(f1), "class_distribution": {i: np.sum(predictions == i) for i in range(3)}
        # Skip class_distribution which is a dictionary
    }

# Function to predict sentiment
def predict_sentiment(model, tokenizer, aspect, sentence):
    """
    Predicts sentiment for a given aspect-sentence pair.
    
    Args:
        model: Fine-tuned sentiment classification model
        tokenizer: Corresponding tokenizer for the model  
        aspect: Target aspect term (str)
        sentence: Full input sentence (str)
    
    Returns:
        int: Predicted class index (0=negative, 1=neutral, 2=positive)
    
    Process:
    1. Formats input with aspect and sentence
    2. Tokenizes and prepares input tensors
    3. Runs model inference (no gradients)
    4. Returns class with highest probability
    """
    input_text = f"Aspect: {aspect} | Sentence: {sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Ensure input tensors are on the same device as the model (cuda or cpu)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()  # Get predicted class
    return prediction

def absa_model(data, model_name, rn1, rn2, epochs, save = False, early_stopping_patience=3):
    """
    Trains and evaluates an Aspect-Based Sentiment Analysis (ABSA) model.
    
    Args:
        data: DataFrame containing raw text, aspect terms, and polarities
        model_name: Pretrained model identifier (e.g., "bert-base-german")
        rn1: Random seed for train/validation split
        rn2: Random seed for validation/test split  
        epochs: Number of training epochs
        save: Whether to save the trained model (default: False)
        early_stopping_patience: Epochs to wait before early stopping (default: 3)
    
    Returns:
        None (prints evaluation metrics and saves artifacts)
    
    Key Operations:
    1. Data Preparation:
       - Splits data into train/val/test sets (80/10/10)
       - Formats aspect-sentence pairs
       - Handles class imbalance via weighting
    
    2. Model Training:
       - Initializes specified transformer model
       - Configures training with custom callbacks
       - Applies relative early stopping
    
    3. Evaluation:
       - Generates classification reports
       - Produces confusion matrices (raw and normalized)
       - Saves predictions and model artifacts
    
    Outputs:
    - Console metrics (F1, precision, recall)
    - Saved model files (if save=True)
    - Test predictions in text files
    - Visualization plots
    """
    
    # Convert aspectTerms column from JSON-like strings to Python lists of dictionaries
    data["aspectTerms"] = data["aspectTerms"].apply(lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x)
    
    # First, split 80% for training and 20% for temporary (validation + test)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=rn1)
    # Then split the temporary set into 50% validation and 50% test (10% each of the original data)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=rn2)
    
    # Initialize lists to store all datasets
    datasets = []
    dataset_names = ["Training", "Validation", "Test"]
    
    # Process each dataset separately
    for data_split, name in zip([train_data, val_data, test_data], dataset_names):
        aspect_sentences = []
        labels = []
        sentiment_mapping = {"negativ": 0, "neutral": 1, "positiv": 2}  # Map sentiment to labels
    
        for _, row in data_split.iterrows():
            for aspect in row["aspectTerms"]:
                if aspect["term"] != "noAspectTerm":
                    aspect_sentences.append(f"Sentence: {row['raw_text']} Aspect: {aspect['term']}")
                    labels.append(sentiment_mapping[aspect["polarity"]])
        
        # Create a DataFrame for this split
        split_df = pd.DataFrame({"text": aspect_sentences, "labels": labels})
        datasets.append(split_df)
        
        # Count the labels
        label_counts = split_df["labels"].value_counts()
        # Map the sentiment names back to the counts
        sentiment_counts = {sentiment: label_counts.get(label, 0) for sentiment, label in sentiment_mapping.items()}
        print(f"{name} Sentiment label count: ", sentiment_counts)
    
    # Now create Hugging Face datasets
    train_dataset = Dataset.from_dict({"text": datasets[0]["text"], "labels": datasets[0]["labels"]})
    val_dataset = Dataset.from_dict({"text": datasets[1]["text"], "labels": datasets[1]["labels"]})
    test_dataset = Dataset.from_dict({"text": datasets[2]["text"], "labels": datasets[2]["labels"]})

    # Compute class weights
    class_weights = compute_class_weight( 'balanced', classes=np.unique(datasets[0]["labels"]), y=datasets[0]["labels"])
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights for (negative, neutral, positive):", class_weights)

    if model_name in ["GerMedBERT/medbert-512", "TUM/GottBERT_base_last","TUM/GottBERT_base_best",
                      "TUM/GottBERT_filtered_base_best", "deepset/gbert-base"]:
        if model_name in ["GerMedBERT/medbert-512", "deepset/gbert-base"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        pipe = pipeline("fill-mask", model=model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type="single_label_classification") 
    else:
        if model_name == "tabularisai/multilingual-sentiment-analysis":
            pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3,problem_type="single_label_classification")

    model.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights) # apply class weights
    
    # Tokenize data
    # Use partial to bind the tokenizer argument
    tokenize_with_tokenizer = partial(tokenize_function, tokenizer)
    
    # Apply the tokenization to each dataset
    train_dataset = train_dataset.map(tokenize_with_tokenizer, batched=True)
    val_dataset = val_dataset.map(tokenize_with_tokenizer, batched=True)
    test_dataset = test_dataset.map(tokenize_with_tokenizer, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./models/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size= 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_steps=500,  # Save every 500 steps instead of 100
        warmup_steps=100,
        warmup_ratio=0.1,
    )

    if model_name in ["FacebookAI/xlm-roberta-base", "aari1995/German_Sentiment", "TUM/GottBERT_filtered_base_best"]:
        training_args.learning_rate = 1e-5
        #training_args.per_device_train_batch_size = 4

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Handles tokenization dynamically

    # Initialize metrics tracker
    metrics = MetricsTracker()
    
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
        #tokenizer=tokenizer, #Deprecated
        data_collator=data_collator,  # Add data collator for proper padding
        compute_metrics=compute_metrics,  # Use custom evaluation function
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], 
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=0.001)], # early stopping to prevent overfitting to one class, continues when there's any improvement
        #callbacks=[RelativeEarlyStopping(patience=early_stopping_patience, min_improvement=0.003)], # early stopping to prevent overfitting to one class, continues when there's any improvement
        callbacks=[MetricsCallback(), RelativeEarlyStopping(patience=early_stopping_patience, min_improvement=0.003)]
    )

    print(f"Training results for {model_name} with {epochs} epochs and random seeds: {rn1}, {rn2}")
    print("")
    # Train the model
    trainer.train()

    if save == True:
        trainer.save_model(f'./saved_models/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        print(f'\nBest Model saved at: ./saved_models/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        tokenizer.save_pretrained(f'./saved_tokenizers/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')
        print(f'\nTokenizer for best Model saved at: ./saved_tokenizers/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}')

    print(f"Evaluation results for {model_name} with {epochs} epochs and random seeds: {rn1}, {rn2}")
    print("")
    
    # Evaluate the model
    eval_results = trainer.evaluate(test_dataset)
    print(eval_results)

    # Prepare the model for inference
    model.eval()  # Set model to evaluation mode
    
    # Collect all predictions and true labels
    true_labels = []
    predictions = []
    
    # Iterate over the test dataset and collect true labels and predicted labels
    for i in range(len(test_dataset)):
        # Extract sentence and aspect from the formatted text
        sentence = test_dataset[i]["text"]
        parts = test_dataset[i]["text"].split(" Aspect: ")
        aspect = parts[1] if len(parts) > 1 else "noAspectTerm"
        true_label = test_dataset[i]["labels"]
        
        # Append the true label to the list
        true_labels.append(true_label)
        
        # Get the predicted sentiment
        predicted_sentiment = predict_sentiment(model, tokenizer, aspect, sentence)
        
        # Append the predicted label to the list
        predictions.append(predicted_sentiment)
    
    # Generate the classification report
    report = classification_report(true_labels, predictions, target_names=["Negativ", "Neutral", "Positiv"], zero_division=0)
    
    # Print the classification report
    print(report)
    print("True label distribution:", {i: true_labels.count(i) for i in range(3)})
    print("Predicted label distribution:", {i: predictions.count(i) for i in range(3)})
    
    # Optionally, save the classification report to a text file
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Generate the classification report as a dictionary
    report_dict = classification_report(true_labels, predictions, target_names=["Negativ", "Neutral", "Positiv"], output_dict=True, zero_division=0)
    
    # Print Precision, Recall, and F1 for each class
    for class_name in ["Negativ", "Neutral", "Positiv"]:
        print(f"{class_name} Precision Score: {report_dict[class_name]['precision']}")
        print(f"{class_name} Recall Score: {report_dict[class_name]['recall']}")
        print(f"{class_name} F1 Score: {report_dict[class_name]['f1-score']}")
        print()
    
    # Print the Macro Average Precision, Recall, and F1
    print(f"Macro Average Precision Score: {report_dict['macro avg']['precision']}")
    print(f"Macro Average Recall Score: {report_dict['macro avg']['recall']}")
    print(f"Macro Average F1 Score: {report_dict['macro avg']['f1-score']}")

    # Print the Weighted Average Precision, Recall, and F1
    print(f"\nWeighted Average Precision: {report_dict['weighted avg']['precision']}")
    print(f"Weighted Average Recall: {report_dict['weighted avg']['recall']}")
    print(f"Weighted Average F1: {report_dict['weighted avg']['f1-score']}")

    # Prepare the model for inference
    model.eval()  # Set model to evaluation mode

    # Open a file to save the results
    result_file = f'testresult/BO/absa/{epochs}_epochs/{model_name.replace("/", "_")}_absa_test_results.txt'

    # Create the folder path if it doesn't exist
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file , "w") as f:
        for i in range(len(test_dataset)):
            # Extract sentence and aspect from the formatted text
            sentence = test_dataset[i]["text"]
            parts = test_dataset[i]["text"].split(" Aspect: ")
            aspect = parts[1] if len(parts) > 1 else "noAspectTerm"
            true_sentiment = test_dataset[i]["labels"]

            # Predict sentiment for the sentence
            predicted_sentiment = predict_sentiment(model, tokenizer, aspect, sentence)
            
            # Write the results to the file
            f.write(f"{sentence}\n")
            f.write(f"True Sentiment: {true_sentiment}\n")
            f.write(f"Predicted Sentiment: {predicted_sentiment}\n")
            f.write("-" * 50 + "\n")  # Separator line for clarity

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    # Class names
    class_names = ["Negativ", "Neutral", "Positiv"]
    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    # Save Confusion Matrix as PNG
    confusion_matrix_file = f'testresult/BO/absa/{epochs}_epochs/{model_name.replace("/", "_")}_confusion_matrix.png'
    plt.savefig(confusion_matrix_file)
    print(f"\nConfusion Matrix saved at: {confusion_matrix_file}")
    # Optional: Show the plot
    # plt.show()
    plt.close()

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions, normalize="true")
    # Class names
    class_names = ["Negativ", "Neutral", "Positiv"]
    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    # Save Confusion Matrix as PNG
    confusion_matrix_file = f'testresult/BO/absa/{epochs}_epochs/{model_name.replace("/", "_")}_confusion_matrix_normalized.png'
    plt.savefig(confusion_matrix_file)
    print(f"Normalized Confusion Matrix saved at: {confusion_matrix_file}")
    # Optional: Show the plot
    # plt.show()
    plt.close()

    # Print performance metrics
    performance_metrics = metrics.get_metrics()
    print("\n=== Performance Metrics ===")
    print(f"GPU: {performance_metrics['gpu_type']}")
    print(f"Average epoch time: {performance_metrics['avg_epoch_time']:.2f}s")
    print(f"Total training time: {performance_metrics['total_train_time']:.2f}s")
    print(f"Peak GPU memory: {performance_metrics['peak_memory']}MB")
    print(f"Average batch time: {performance_metrics['avg_batch_time']:.4f}s")
    
    # Save metrics
    os.makedirs(f"performanceresults/absa/{epochs}_epochs", exist_ok=True)
    with open(f"performanceresults/absa/{epochs}_epochs/performance_{model_name.replace('/','_')}.json", "w") as f:
        json.dump(performance_metrics, f, indent=2)

    # Delete the saved model directory to free disk space
    output_dir= f'./models/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}'
    shutil.rmtree(output_dir, ignore_errors=True)
    print("\nTraining complete. Model directory deleted to free memory.")

def absa_model_kfold(data, model_name, rn1, rn2, epochs, n_splits=5, save=False, early_stopping_patience=3):
    """
    Performs k-fold cross validation for ABSA model training and evaluation.
    
    Args:
        data: DataFrame containing text, aspect terms and polarities
        model_name: Pretrained transformer model identifier
        rn1: Random seed for k-fold splitting
        rn2: Random seed for validation/test splitting
        epochs: Number of training epochs per fold
        n_splits: Number of cross-validation folds (default: 5)
        save: Whether to save best model from each fold (default: False)
        early_stopping_patience: Epochs to wait before early stopping (default: 3)
    
    Returns:
        tuple: (avg_metrics, std_metrics) dictionaries containing:
        - Mean and standard deviation of evaluation metrics across folds
        - Keys: avg_eval_accuracy, std_eval_accuracy, etc.
    
    Key Features:
    - Stratified k-fold cross validation
    - Class-weighted loss for imbalanced data
    - Per-fold and aggregated metrics
    - Saves models, reports and visualizations
    - Includes normalized and raw confusion matrices
    - Handles multiple model architectures
    
    Key Operations:
    1. Data Preparation:
       - K-fold splitting with stratification
       - Aspect-sentence pair formatting
       - Automatic class weight calculation
    
    2. Model Training:
       - Custom training per fold
       - Relative early stopping
       - Model checkpointing
    
    3. Evaluation:
       - Per-fold and aggregated metrics
       - Detailed classification reports
       - Visualizations of results
    """
    
    # Convert aspectTerms column from JSON-like strings to Python lists of dictionaries
    data["aspectTerms"] = data["aspectTerms"].apply(lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rn1)
    
    # Initialize fold_metrics as a regular dict with predefined keys
    fold_metrics = {
        'eval_accuracy': [],
        'eval_precision': [],
        'eval_recall': [],
        'eval_f1': [],
        'eval_loss': []
    }
    all_true_labels = []
    all_predictions = []
    
    # Prepare output directories
    os.makedirs(f'testresult/BO/absa/{epochs}_epochs/kfold/', exist_ok=True)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold}/{n_splits}")
        print(f"{'='*50}\n")
        
        # Split data into train and test for this fold
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Further split test into validation and test (50/50)
        test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=rn2)
        
        # Process each dataset
        datasets = []
        for data_split, name in zip([train_data, val_data, test_data], ["Training", "Validation", "Test"]):
            aspect_sentences = []
            labels = []
            sentiment_mapping = {"negativ": 0, "neutral": 1, "positiv": 2}
            
            for _, row in data_split.iterrows():
                for aspect in row["aspectTerms"]:
                    if aspect["term"] != "noAspectTerm":
                        aspect_sentences.append(f"Sentence: {row['raw_text']} Aspect: {aspect['term']}")
                        labels.append(sentiment_mapping[aspect["polarity"]])
            
            split_df = pd.DataFrame({"text": aspect_sentences, "labels": labels})
            datasets.append(split_df)
            
            # Print label distribution
            label_counts = split_df["labels"].value_counts()
            sentiment_counts = {sentiment: label_counts.get(label, 0) for sentiment, label in sentiment_mapping.items()}
            print(f"{name} Fold {fold} Sentiment label count: ", sentiment_counts)
        
        # Create Hugging Face datasets
        train_dataset = Dataset.from_dict({"text": datasets[0]["text"], "labels": datasets[0]["labels"]})
        val_dataset = Dataset.from_dict({"text": datasets[1]["text"], "labels": datasets[1]["labels"]})
        test_dataset = Dataset.from_dict({"text": datasets[2]["text"], "labels": datasets[2]["labels"]})

        # Compute class weights
        class_weights = compute_class_weight( 'balanced', classes=np.unique(datasets[0]["labels"]), y=datasets[0]["labels"])
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        print("Class weights for (negative, neutral, positive):", class_weights)
        
        # Load tokenizer and model
        if model_name in ["GerMedBERT/medbert-512", "TUM/GottBERT_base_last","TUM/GottBERT_base_best",
                          "TUM/GottBERT_filtered_base_best", "deepset/gbert-base"]:
            if model_name in ["GerMedBERT/medbert-512", "deepset/gbert-base"]:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
            pipe = pipeline("fill-mask", model=model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type="single_label_classification") 
        else:
            if model_name == "tabularisai/multilingual-sentiment-analysis":
                pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type="single_label_classification") 

        model.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights) # apply class weights

        # Tokenize data
        tokenize_with_tokenizer = partial(tokenize_function, tokenizer)
        train_dataset = train_dataset.map(tokenize_with_tokenizer, batched=True)
        val_dataset = val_dataset.map(tokenize_with_tokenizer, batched=True)
        test_dataset = test_dataset.map(tokenize_with_tokenizer, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./models/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}_fold{fold}',
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_steps=500,  # Save every 500 steps instead of 100
            warmup_steps=100,
            warmup_ratio=0.1,
        )

        if model_name in ["FacebookAI/xlm-roberta-base", "aari1995/German_Sentiment", "TUM/GottBERT_filtered_base_best"]:
            training_args.learning_rate = 1e-5
            #training_args.per_device_train_batch_size = 4

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Handles tokenization dynamically

        # Initialize metrics for this fold
        metrics = MetricsTracker()
        
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
            #tokenizer=tokenizer, # Deprecated
            data_collator=data_collator,
            compute_metrics=compute_metrics_kfold,
            #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], 
            #callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=0.001)], # early stopping to prevent overfitting to one class, continues when there's any improvement
            #callbacks=[RelativeEarlyStopping(patience=early_stopping_patience, min_improvement=0.003)], # early stopping to prevent overfitting to one class, continues when there's any improvement
            callbacks=[MetricsCallback(), RelativeEarlyStopping(patience=early_stopping_patience, min_improvement=0.003)]
        )
        
        # Train the model
        trainer.train()
        
        if save:
            trainer.save_model(f'./saved_models/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}_fold{fold}')
            tokenizer.save_pretrained(f'./saved_tokenizers/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}_fold{fold}')
        
        # Evaluate on test set
        eval_results = trainer.evaluate(test_dataset)
        
        # Store only numerical metrics for this fold
        for metric in fold_metrics.keys():
            if metric in eval_results:
                fold_metrics[metric].append(eval_results[metric])
        
        # Collect predictions and true labels
        model.eval()
        true_labels = []
        predictions = []
        
        for i in range(len(test_dataset)):
            # Extract sentence and aspect from the formatted text
            sentence = test_dataset[i]["text"]
            parts = test_dataset[i]["text"].split(" Aspect: ")
            aspect = parts[1] if len(parts) > 1 else "noAspectTerm"
            true_label = test_dataset[i]["labels"]
            true_labels.append(true_label)
            predicted_sentiment = predict_sentiment(model, tokenizer, aspect, sentence)
            predictions.append(predicted_sentiment)
        
        # Add to global lists for overall evaluation
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
        
        # Generate classification report for this fold
        report = classification_report(true_labels, predictions, target_names=["Negativ", "Neutral", "Positiv"], zero_division=0)
        print(f"\nClassification Report for Fold {fold}:")
        print(report)
        
        # Save fold results
        with open(f'testresult/BO/absa/{epochs}_epochs/kfold/{model_name.replace("/", "_")}_fold{fold}_results.txt', "w") as f:
            f.write(f"Fold {fold} Evaluation Results:\n")
            f.write(json.dumps(eval_results, indent=2))
            f.write("\n\nClassification Report:\n")
            f.write(report)

        report_dict = classification_report(
            true_labels, predictions, 
            target_names=["Negativ", "Neutral", "Positiv"], 
            output_dict=True, zero_division=0
        )
        with open(f'testresult/BO/absa/{epochs}_epochs/kfold/{model_name.replace("/", "_")}_fold{fold}_results.txt', "a") as f:
            f.write(f"\nMacro Average Precision: {report_dict['macro avg']['precision']}\n")
            f.write(f"Macro Average Recall: {report_dict['macro avg']['recall']}\n")
            f.write(f"Macro Average F1: {report_dict['macro avg']['f1-score']}\n")
            f.write(f"\nWeighted Average Precision: {report_dict['weighted avg']['precision']}\n")
            f.write(f"Weighted Average Recall: {report_dict['weighted avg']['recall']}\n")
            f.write(f"Weighted Average F1: {report_dict['weighted avg']['f1-score']}\n")

        # Get and store fold metrics
        fold_performance = metrics.get_metrics()
        print(f"\nFold {fold} Performance Metrics:")
        print(f"GPU: {fold_performance['gpu_type']}")
        print(f"Avg epoch: {fold_performance['avg_epoch_time']:.2f}s")
        print(f"Total: {fold_performance['total_train_time']:.2f}s")
        print(f"Peak memory: {fold_performance['peak_memory']}MB")
        print(f"Avg batch: {fold_performance['avg_batch_time']:.4f}s")
        
        # Save fold metrics
        os.makedirs(f"performanceresults/absa/{epochs}_epochs/kfold/", exist_ok=True)
        with open(f"performanceresults/absa/{epochs}_epochs/kfold/{model_name.replace("/", "_")}_fold{fold}_performance.json", "w") as f:
            json.dump(fold_performance, f, indent=2)
        
        # Clean up
        output_dir = f'./models/absa_{model_name.replace("/", "_")}_{rn1}_{rn2}_{epochs}_fold{fold}'
        shutil.rmtree(output_dir, ignore_errors=True)
    
    # Calculate average metrics across all folds
    avg_metrics = {}
    std_metrics = {}
    
    for metric, values in fold_metrics.items():
        if len(values) > 0:  # Only process metrics that have values
            avg_metrics[f"avg_{metric}"] = np.mean(values)
            std_metrics[f"std_{metric}"] = np.std(values)
    
    print("\n" + "="*50)
    print("Cross-Validation Summary")
    print("="*50)
    print("\nAverage Metrics Across Folds:")
    for metric, values in fold_metrics.items():
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Generate overall classification report
    overall_report = classification_report(all_true_labels, all_predictions, 
                                        target_names=["Negativ", "Neutral", "Positiv"],
                                        zero_division=0)
    print("\nOverall Classification Report:")
    print(overall_report)

    # Add after the overall report (around line 600)
    overall_report_dict = classification_report(
        all_true_labels, all_predictions,
        target_names=["Negativ", "Neutral", "Positiv"],
        output_dict=True, zero_division=0
    )    

    print("\nWeighted Average Scores Across All Folds:")
    print(f"Precision: {overall_report_dict['weighted avg']['precision']:.4f}")
    print(f"Recall: {overall_report_dict['weighted avg']['recall']:.4f}")
    print(f"F1: {overall_report_dict['weighted avg']['f1-score']:.4f}")
    
    # Save overall results
    with open(f'testresult/BO/absa/{epochs}_epochs/kfold/{model_name.replace("/", "_")}_overall_results.txt', "w") as f:
        f.write("Cross-Validation Summary\n")
        f.write("="*50 + "\n")
        f.write("\nAverage Metrics Across Folds:\n")
        for metric, values in fold_metrics.items():
            f.write(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        f.write("\nWeighted Average Scores Across All Folds:\n")
        f.write(f"Precision: {overall_report_dict['weighted avg']['precision']:.4f}\n")
        f.write(f"Recall: {overall_report_dict['weighted avg']['recall']:.4f}\n")
        f.write(f"F1: {overall_report_dict['weighted avg']['f1-score']:.4f}\n")
        f.write("\nOverall Classification Report:\n")
        f.write(overall_report)       
    
    # Generate overall confusion matrices
    class_names = ["Negativ", "Neutral", "Positiv"]
    
    # Regular confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'testresult/BO/absa/{epochs}_epochs/kfold/{model_name.replace("/", "_")}_overall_confusion_matrix.png')
    plt.close()
    
    # Normalized confusion matrix
    cm_norm = confusion_matrix(all_true_labels, all_predictions, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'testresult/BO/absa/{epochs}_epochs/kfold/{model_name.replace("/", "_")}_overall_confusion_matrix_normalized.png')
    plt.close()
    
    return avg_metrics, std_metrics

