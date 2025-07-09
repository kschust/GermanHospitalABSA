from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, RobertaTokenizerFast, AutoModelForMaskedLM, pipeline
import torch
import json
import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
import re
from collections import defaultdict, Counter
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from difflib import SequenceMatcher

# --- FUNCTIONS FOR ATE CAT ---

# Define the label to category mapping
LABEL_TO_CATEGORY = {
    'LABEL_0': None,  # Not an aspect term
    'LABEL_1': 'Allgemein',
    'LABEL_2': 'anderer Service',
    'LABEL_3': 'Arzt',
    'LABEL_4': 'Krankenhaus',
    'LABEL_5': 'mediz. Service',
    'LABEL_6': 'Pflegepersonal',
    'LABEL_7': 'Personal'
}

def extract_aspects_with_categories(text, ate_pipeline):
    """
    Extract aspect terms and their categories from the given text using the ATE pipeline.
    
    Args:
        text: The input text to analyze
        ate_pipeline: Aspect Term Extraction pipeline
        
    Returns:
        JSON string containing list of aspect dictionaries with 'term' and 'category' fields
    """
    
    # Get model predictions
    predictions = ate_pipeline(text)
    
    # Sort predictions by their start position
    predictions = sorted(predictions, key=lambda x: x['start'])
    
    aspects = []
    i = 0
    n = len(predictions)
    
    while i < n:
        pred = predictions[i]
        
        # Only process non-LABEL_0 predictions
        if pred['entity_group'] != 'LABEL_0':
            # Get the corresponding category
            category = LABEL_TO_CATEGORY.get(pred['entity_group'], 'Allgemein')
            
            # Initialize term with current prediction
            aspect_term = pred['word']
            start_pos = pred['start']
            end_pos = pred['end']

            # Remove leading space if present
            if aspect_term.startswith(' '):
                aspect_term = aspect_term[1:]
                start_pos += 1  # Adjust start position
            
            # Handle ##-prefixed terms (needing characters from previous word)
            if aspect_term.startswith('##') and i > 0:
                prev_pred = predictions[i-1]
                if prev_pred['entity_group'] == 'LABEL_0':
                    # Get the full previous word text
                    prev_text = text[prev_pred['start']:prev_pred['end']]
                    
                    # Take all characters after last space in previous word
                    last_space_pos = prev_text.rfind(' ')
                    prefix = prev_text[last_space_pos+1:] if last_space_pos >= 0 else prev_text
                    
                    # Combine prefix with term (without ##)
                    aspect_term = prefix + aspect_term[2:]
                    start_pos = prev_pred['end'] - len(prefix)

            # Check the next token if it's a continuation (LABEL_0 with ##)
            if i + 1 < n:
                next_pred = predictions[i+1]
                if (next_pred['entity_group'] == 'LABEL_0' and 
                    next_pred['word'].startswith('##')):
                    
                    # Extract the continuation part (before first space)
                    continuation = next_pred['word'][2:].split()[0]  # Take before space
                    aspect_term += continuation
                    end_pos = next_pred['start'] + len(continuation)
            
            # Add to aspects list with category
            aspects.append({
                'term': aspect_term,
                'category': category
            })
        
        i += 1
    
    # Return "Allgemein" if no aspects found
    if not aspects:
        return json.dumps([{'term': 'noAspectTerm', 'category': 'Allgemein'}], ensure_ascii=False)
    
    return json.dumps(aspects, ensure_ascii=False)

def extract_aspect_entries(entry):
    """
    Extract and normalize aspect entries from various formats into a consistent structure.
    
    Args:
        entry: Aspect data in string, list, or dict format
        
    Returns:
        List of normalized aspect dictionaries with 'term' and 'category' fields
    """
    try:
        if isinstance(entry, str):
            if entry.startswith('['):
                aspects = ast.literal_eval(entry)
            else:
                aspects = json.loads(entry)
        else:
            aspects = entry
        
        if isinstance(aspects, list):
            normalized = []
            for item in aspects:
                if isinstance(item, dict):
                    # Handle both term+polarity and category+polarity formats
                    if 'term' in item:
                        normalized.append({
                            'term': item['term'],
                            'category': item.get('category', 'Allgemein')
                        })
                    elif 'category' in item:
                        # This is from aspectCategories column - needs to be paired with terms
                        normalized.append({
                            'category': item['category'],
                            'term': None  # Will be filled when pairing
                        })
                elif isinstance(item, str):
                    normalized.append({
                        'term': item,
                        'category': 'Allgemein'
                    })
            return normalized
        return []
    except:
        return []

def is_partial_match(pred_term, true_term, threshold=0.7):
    """
    Check if two terms are partial matches based on similarity threshold.
    
    Args:
        pred_term: Predicted term
        true_term: Ground truth term
        threshold: Similarity threshold (0-1)
        
    Returns:
        Boolean indicating whether terms are considered a match
    """
    # Case-insensitive comparison
    pred_lower = pred_term.lower()
    true_lower = true_term.lower()
    
    # Check for substring match (either direction)
    if pred_lower in true_lower or true_lower in pred_lower:
        return True
    
    # Check similarity ratio
    return SequenceMatcher(None, pred_lower, true_lower).ratio() >= threshold

def evaluate_with_partial_matches_cat(df, true_terms_col='aspectTerms', true_cat_col='aspectCategories', pred_col='predicted_aspects_with_cat', partial_threshold=0.7):
    """
    Evaluate aspect terms and categories with support for partial term matching.
    
    Args:
        df: DataFrame containing the data
        true_terms_col: Column name for true aspect terms
        true_cat_col: Column name for true aspect categories
        pred_col: Column name for predicted aspects
        partial_threshold: Similarity threshold for partial matches
        
    Returns:
        Dictionary containing exact and partial match metrics at term, category, and combined levels
    """
    results = {
        'exact': {
            'term': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0},
            'category': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0},
            'combined': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        },
        'partial': {
            'term': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0},
            'category': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0},
            'combined': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        },
        'errors': {
            'exact': {'false_positives': defaultdict(list), 'false_negatives': defaultdict(list), 'true_negatives': defaultdict(list) },
            'partial': {'false_positives': defaultdict(list), 'false_negatives': defaultdict(list), 'true_negatives': defaultdict(list) },
            'partial_matches': defaultdict(list)
        }
    }

    for idx, row in df.iterrows():
        # Extract true and predicted aspects
        true_terms = extract_aspect_entries(row[true_terms_col])
        true_cats = extract_aspect_entries(row[true_cat_col])
        pred_aspects = extract_aspect_entries(row[pred_col])
        
        # Pair true terms with categories
        true_aspects = []
        for i, term in enumerate(true_terms):
            cat = true_cats[i]['category'] if i < len(true_cats) else 'Allgemein'
            true_aspects.append({'term': term['term'], 'category': cat})
        
        # Handle 'noAspectTerm' cases
        is_true_negative = len(true_aspects) == 1 and true_aspects[0]['term'] == 'noAspectTerm'
        is_predicted_negative = len(pred_aspects) == 1 and pred_aspects[0]['term'] == 'noAspectTerm'
        
        # Case 1: True Negative (correctly predicted no aspects)
        if is_true_negative and is_predicted_negative:
            for match_type in ['exact', 'partial']:
                for metric in ['term', 'category', 'combined']:
                    results[match_type][metric]['tn'] += 1
                    results['errors'][match_type]['true_negatives'][row['sentence_id']].append(row['raw_text'])
            continue
        
        # Case 2: False Positive (predicted aspects when shouldn't)
        if is_true_negative and not is_predicted_negative:
            num_fp = len(pred_aspects)
            for match_type in ['exact', 'partial']:
                for metric in ['term', 'category', 'combined']:
                    results[match_type][metric]['fp'] += num_fp
                for aspect in pred_aspects:
                    results['errors'][match_type]['false_positives'][aspect['term']].append(row['raw_text'])
            continue
        
        # Case 3: False Negative (missed true aspects)
        if not is_true_negative and is_predicted_negative:
            num_fn = len(true_aspects)
            for match_type in ['exact', 'partial']:
                for metric in ['term', 'category', 'combined']:
                    results[match_type][metric]['fn'] += num_fn
                for aspect in true_aspects:
                    results['errors'][match_type]['false_negatives'][aspect['term']].append(row['raw_text'])
            continue
        
        # Standard case: Compare aspects
        true_terms = [a for a in true_aspects if a['term'] != 'noAspectTerm']
        pred_terms = [a for a in pred_aspects if a['term'] != 'noAspectTerm']
        
        # Track matches
        exact_term_matches = set()
        partial_term_matches = set()
        exact_category_matches = set()
        partial_category_matches = set()
        
        for pred in pred_terms:
            exact_match = False
            partial_match = False
            
            for true in true_terms:
                # Exact term match
                if pred['term'] == true['term']:
                    exact_term_matches.add((pred['term'], true['term']))
                    exact_match = True
                    
                    # Exact category match
                    if pred['category'] == true['category']:
                        exact_category_matches.add((pred['term'], true['term']))
                    break
                
                # Partial term match
                elif is_partial_match(pred['term'], true['term'], partial_threshold):
                    partial_term_matches.add((pred['term'], true['term']))
                    partial_match = True
                    results['errors']['partial_matches'][f"{pred['term']} → {true['term']}"].append(row['raw_text'])
                    
                    # Partial category match
                    if pred['category'] == true['category']:
                        partial_category_matches.add((pred['term'], true['term']))
                    break
            
            # Record false positives
            if not exact_match:
                results['errors']['exact']['false_positives'][pred['term']].append(row['raw_text'])
            if not (exact_match or partial_match):
                results['errors']['partial']['false_positives'][pred['term']].append(row['raw_text'])
        
        # Update counts
        results['exact']['term']['tp'] += len(exact_term_matches)
        results['partial']['term']['tp'] += len(exact_term_matches) + len(partial_term_matches)
        
        results['exact']['category']['tp'] += len(exact_category_matches)
        results['partial']['category']['tp'] += len(exact_category_matches) + len(partial_category_matches)
        
        results['exact']['combined']['tp'] += len([m for m in exact_category_matches if m in exact_term_matches])
        results['partial']['combined']['tp'] += len([m for m in exact_category_matches if m in exact_term_matches]) + \
                                             len([m for m in partial_category_matches if m in partial_term_matches])
        
        # Record false negatives
        for true in true_terms:
            exact_matched = any(true['term'] == m[1] for m in exact_term_matches)
            partial_matched = any(is_partial_match(m[1], true['term'], partial_threshold) for m in partial_term_matches)
            
            if not exact_matched:
                results['errors']['exact']['false_negatives'][true['term']].append(row['raw_text'])
                results['exact']['term']['fn'] += 1
                results['exact']['category']['fn'] += 1
                results['exact']['combined']['fn'] += 1
            
            if not (exact_matched or partial_matched):
                results['errors']['partial']['false_negatives'][true['term']].append(row['raw_text'])
                results['partial']['term']['fn'] += 1
                results['partial']['category']['fn'] += 1
                results['partial']['combined']['fn'] += 1

    # Calculate metrics
    for match_type in ['exact', 'partial']:
        for metric in ['term', 'category', 'combined']:
            tp = results[match_type][metric]['tp']
            tn = results[match_type][metric]['tn']
            fp = results[match_type][metric]['fp']
            fn = results[match_type][metric]['fn']
            
            # Precision
            if tp + fp > 0:
                results[match_type][metric]['precision'] = tp / (tp + fp)
            
            # Recall
            if tp + fn > 0:
                results[match_type][metric]['recall'] = tp / (tp + fn)
            
            # F1
            if results[match_type][metric]['precision'] + results[match_type][metric]['recall'] > 0:
                results[match_type][metric]['f1'] = 2 * (
                    results[match_type][metric]['precision'] * results[match_type][metric]['recall']
                ) / (
                    results[match_type][metric]['precision'] + results[match_type][metric]['recall']
                )
            
            # Accuracy
            if tp + tn + fp + fn > 0:
                results[match_type][metric]['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    return results

def print_metrics(metrics, title, level='term', match_type='exact'):
    """
    Print metrics for the new evaluation format with TN support.
    
    Args:
        metrics: The results dictionary from evaluate_with_tn_handling()
        title: Header for the output
        level: 'term', 'category', or 'combined'
        match_type: 'exact' or 'partial'
    """
    # Get the relevant metrics subset
    m = metrics[match_type][level]
    
    print(f"\n=== {title} ({match_type} match, {level} level) ===")
    
    # Calculate confusion matrix values
    tp = m['tp']
    tn = m['tn']
    fp = m['fp']
    fn = m['fn']
    
    # Print standard metrics
    print(f"Precision: {m['precision']:.4f} ({tp}/{tp + fp})")
    print(f"Recall:    {m['recall']:.4f} ({tp}/{tp + fn})")
    print(f"F1:        {m['f1']:.4f}")
    print(f"Accuracy:  {m['accuracy']:.4f} ({(tp + tn)}/{(tp + tn + fp + fn)})")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("               Predicted")
    print("               Aspect  No Aspect")
    print(f"Actual Aspect  {tp:6}  {fn:6}")
    print(f"No Aspect      {fp:6}  {tn:6}")

def print_cat_evaluation(df, results, max_examples=5):
    """
    Print comprehensive evaluation results for aspect term extraction and categorization,
    including exact/partial matches, error analysis, and example cases.
    
    Args:
        df: DataFrame containing the evaluation data
        results: Dictionary containing evaluation metrics and error analysis from evaluate_with_partial_matches_cat()
        max_examples: Maximum number of examples to show for each error type (default: 5)
        
    Outputs:
        - Exact and partial match metrics for terms and term+category combinations
        - Partial match examples (top 5 most frequent)
        - False positives/negatives analysis (top 5 most frequent for each match type)
        - True negative examples (up to max_examples)
        - False negative examples (up to max_examples, if available)
    """
    print("=== EXACT MATCHES ===")
    print_metrics(results, "Terms", level='term', match_type='exact')
    print_metrics(results, "Combined (exact term + category)", level='combined', match_type='exact')

    print("\n=== PARTIAL MATCHES (including exact) ===")
    print_metrics(results, "Terms", level='term', match_type='partial')
    print_metrics(results, "Combined (partial term + category)", level='combined', match_type='partial')

    # Error analysis
    if results['errors']['partial_matches']:
        print("\n=== PARTIAL MATCH EXAMPLES ===")
        for match, count in sorted(
            [(k, len(v)) for k, v in results['errors']['partial_matches'].items()],
            key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"- {match}: {count} occurrences")

    # Print most common errors
    for match_type in ['exact', 'partial']:
        print(f"\n=== {match_type.upper()} FALSE POSITIVES ===")
        for term, count in sorted(
            [(k, len(v)) for k, v in results['errors'][match_type]['false_positives'].items()],
            key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"- {term}: {count}")

        print(f"\n=== {match_type.upper()} FALSE NEGATIVES ===")
        for term, count in sorted(
            [(k, len(v)) for k, v in results['errors'][match_type]['false_negatives'].items()],
            key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"- {term}: {count}")

    # Print true negatives examples
    if any(results['errors']['exact']['true_negatives'].values()):
        print(f"\n=== TRUE NEGATIVE EXAMPLES (first {max_examples}) ===")
        for i, (sent_id, texts) in enumerate(results['errors']['exact']['true_negatives'].items()):
            if i >= max_examples:
                break
            print(f"\n{i+1}. Sentence ID: {sent_id}")
            print(f"   Text: {texts[0]}")

    # Print false negative examples (from false_negs or false_negatives)
    if results['errors'].get('false_negs'):
        print(f"\n=== FALSE NEGATIVE EXAMPLES (first {max_examples}) ===")
        for i, fn in enumerate(results['errors']['false_negs'][:max_examples]):
            print(f"\n{i+1}. Sentence: {fn['sentence']}")
            print(f"   True aspect: {fn['true_aspect']} ({fn['true_category']})")
            print(f"   Predicted: {fn['predicted_aspects']}")

def get_confusion_matrix_cat_data(df, match_type="exact", eval_level="term", threshold=0.7):
    """
    Prepare data for confusion matrix calculation with TN handling.
    
    Args:
        df: DataFrame containing the data
        match_type: 'exact' or 'partial' matching
        eval_level: 'term' or 'combined' evaluation
        threshold: Similarity threshold for partial matches
        
    Returns:
        Tuple of (y_true, y_pred) arrays for confusion matrix
    """    
    y_true = []
    y_pred = []

    # Initialize global counters
    global_tp = global_fp = global_fn = 0

    # 1. First pass: Calculate TP, FP, FN for all documents
    for idx, row in df.iterrows():
        true_terms = extract_aspect_entries(row['aspectTerms'])
        true_cats = extract_aspect_entries(row['aspectCategories'])
        pred_aspects = extract_aspect_entries(row['predicted_aspects_with_cat'])
        
        # Pair true terms with categories
        true_aspects = []
        for i, term in enumerate(true_terms):
            cat = true_cats[i]['category'] if i < len(true_cats) else 'Allgemein'
            true_aspects.append({'term': term['term'], 'category': cat})
        
        true_aspects = [a for a in true_aspects if a['term'] != 'noAspectTerm']
        pred_aspects = [a for a in pred_aspects if a['term'] != 'noAspectTerm']
        
        if match_type == "partial":
            matched_true = set()
            matched_pred = set()
            
            # First check predictions against true aspects
            for pred in pred_aspects:
                for true in true_aspects:
                    if is_partial_match(pred['term'], true['term'], threshold):
                        if eval_level == "term" or pred['category'] == true['category']:
                            matched_true.add(true['term'])
                            matched_pred.add(pred['term'])
                            break
            
            tp = len(matched_true)
            fn = len(true_aspects) - len(matched_true)
            fp = len(pred_aspects) - len(matched_pred)
        else:
            if eval_level == "term":
                true_terms = {a['term'] for a in true_aspects}
                pred_terms = {a['term'] for a in pred_aspects}
                tp = len(true_terms & pred_terms)
                fn = len(true_terms - pred_terms)
                fp = len(pred_terms - true_terms)
            else:
                true_pairs = {(a['term'], a['category']) for a in true_aspects}
                pred_pairs = {(a['term'], a['category']) for a in pred_aspects}
                tp = len(true_pairs & pred_pairs)
                fn = len(true_pairs - pred_pairs)
                fp = len(pred_pairs - true_pairs)
        
        # Accumulate global counts
        global_tp += tp
        global_fp += fp
        global_fn += fn
        
        # Add document-level cases to arrays
        y_true.extend([1]*tp + [1]*fn + [0]*fp)
        y_pred.extend([1]*tp + [0]*fn + [1]*fp)

    # 2. Calculate total words for TN estimation
    all_words = []
    for text in df['raw_text']:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    total_samples = sum(Counter(all_words).values())

    # 3. Calculate and add True Negatives
    global_tn = total_samples - (global_tp + global_fp + global_fn)
    y_true.extend([0]*global_tn)
    y_pred.extend([0]*global_tn)
    
    # Verification
    assert len(y_true) == len(y_pred) == total_samples
    assert sum(y_true) == global_tp + global_fn
    assert sum(y_pred) == global_tp + global_fp
    
    return y_true, y_pred

def plot_conf_matrix_from_results(results, match_type, eval_level, title):
    """
    Plot confusion matrix directly from evaluation results.
    
    Args:
        results: Evaluation results dictionary from evaluate_with_partial_matches_cat
        match_type: 'exact' or 'partial'
        eval_level: 'term' or 'combined'
        title: Plot title
    """

    # Create directory if needed
    save_dir="./pipeline_results/conf_matrix/ate/"
    os.makedirs(save_dir, exist_ok=True)
    # Generate standardized filename
    filename = f"conf_matrix_{eval_level}_{match_type}_match.png"
    save_path = os.path.join(save_dir, filename)
    
    # Get counts directly from the results
    metrics = results[match_type][eval_level]
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    tn = metrics['tn']
    
    # Create y_true and y_pred arrays
    y_true = [1]*tp + [1]*fn + [0]*fp + [0]*tn
    y_pred = [1]*tp + [0]*fn + [1]*fp + [0]*tn
    
    # Plot the matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Aspect", "noAspectTerm"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.show()

def plot_all_confusion_matrices_from_results(results, threshold=0.7):
    """
    Generate all confusion matrix plots using pre-computed results.
    Updated for TN-aware evaluation structure.
    
    Args:
        results: Evaluation results dictionary from evaluate_with_tn_handling()
        threshold: Similarity threshold (for title only)
    """
    # Plot all matrices (no need for TN calculation as it's now included in results)
    plot_conf_matrix_from_results(results, 'exact', 'term', "Exact Term Match Confusion Matrix")
    #plot_conf_matrix_from_results(results, 'exact', 'category', "Exact Category Match Confusion Matrix")
    plot_conf_matrix_from_results(results, 'exact', 'combined', "Exact Term+Category Match Confusion Matrix")
    
    plot_conf_matrix_from_results(results, 'partial', 'term', f"Partial Term Match Confusion Matrix (Threshold={threshold})")
    #plot_conf_matrix_from_results(results, 'partial', 'category', f"Partial Category Match Confusion Matrix (Threshold={threshold})")
    plot_conf_matrix_from_results(results, 'partial', 'combined', f"Partial Term+Category Match Confusion Matrix (Threshold={threshold})")


# --- FUNCTIONS FOR ABSA ---

def get_label_converter(model):
    """
    Creates a label conversion function that standardizes sentiment labels to German format,
    handling different model output formats and case variations based on model's exact label mapping.
    
    Args:
        model: A HuggingFace sentiment analysis model whose label mapping will be inspected
        
    Returns:
        A function that converts various label formats to standardized German sentiment labels:
        - 'positiv' (positive)
        - 'negativ' (negative) 
        - 'neutral' (neutral)
    """
    label_mapping = model.config.id2label
    #print(f"Model label mapping: {label_mapping}")  # Debug output
    
    # Create a direct mapping from all possible label formats to German labels
    conversion_map = {}
    
    # Handle first model format: {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}
    if all(f'LABEL_{i}' in label_mapping.values() for i in range(3)):
        #print("Detected LABEL_X format model - using specific mapping")
        conversion_map.update({
            0: 'negativ',       # LABEL_0 → negativ
            'LABEL_0': 'negativ',
            'label_0': 'negativ',
            1: 'neutral',      # LABEL_1 → neutral
            'LABEL_1': 'neutral',
            'label_1': 'neutral',
            2: 'positiv',       # LABEL_2 → positiv
            'LABEL_2': 'positiv',
            'label_2': 'positiv'
        })
    # Handle second model format: {0: 'positive', 1: 'neutral', 2: 'negative'}
    elif 'positive' in label_mapping.values():
        #print("Detected positive/negative format model - using direct mapping")
        conversion_map.update({
            0: 'negativ',       # positive → negativ (INVERTED)
            'positive': 'negativ',
            'Positive': 'negativ',
            1: 'neutral',       # neutral → neutral 
            'neutral': 'neutral',
            'Neutral': 'neutral',
            2: 'positiv',       # negative → positiv (INVERTED)
            'negative': 'positiv',
            'Negative': 'positiv'
        })
    else:
        print("Unknown label format - defaulting to neutral")
        conversion_map['default'] = 'neutral'
    
    def convert_label(label):
        """
        Converts any label format to standard German labels.
        
        Args:
            label: Input label to convert (can be string, numeric, or other format)
                   Examples: 
                   - 'LABEL_0', 'positive', 'Negative' (strings)
                   - 0, 1, 2 (numeric indices)
                   
        Returns:
            str: Standardized German sentiment label
        """
        if isinstance(label, str):
            label_key = label.lower() if label.lower() in conversion_map else label
        else:
            label_key = label
        
        return conversion_map.get(label_key, conversion_map.get('default', 'neutral'))
    
    return convert_label

def predict_sentiment_for_aspects(text, aspects, absa_pipeline, absa_model):
    """
    Predict sentiment for each aspect in the given text using the ABSA pipeline.
    
    Args:
        text: The full input text
        aspects: List of aspect dictionaries (from extract_aspects_with_categories)
        absa_pipeline: ABSA sentiment pipeline
        absa_model: ABSA sentiment model
        
    Returns:
        List of aspects with added 'sentiment' field ('positiv', 'negativ', 'neutral')
    """
    if not aspects or (len(aspects) == 1 and aspects[0]['term'] == 'noAspectTerm'):
        return []
    
    # Get the appropriate label converter
    label_converter = get_label_converter(absa_model)
    
    aspects_with_sentiment = []
    
    for aspect in aspects:
        input_text = f"{text} [SEP] {aspect['term']}"
        
        try:
            # Get prediction
            prediction = absa_pipeline(input_text, truncation=True)
            raw_label = prediction[0]['label']
            sentiment = label_converter(raw_label)
            
            aspects_with_sentiment.append({
                'term': aspect['term'],
                'category': aspect['category'],
                'sentiment': sentiment,
                'confidence': prediction[0]['score']
            })
            
        except Exception as e:
            print(f"Error processing aspect '{aspect['term']}': {str(e)}")
            aspects_with_sentiment.append({
                'term': aspect['term'],
                'category': aspect['category'],
                'sentiment': 'neutral',
                'confidence': 0.0
            })
    
    return aspects_with_sentiment

def add_sentiment_predictions(df, absa_pipeline, absa_model, text_col='raw_text', aspects_col='predicted_aspects_with_cat'):
    """
    Adds sentiment predictions to aspect terms in a DataFrame.
    
    Args:
        df: Input DataFrame containing text and aspects
        absa_pipeline: ABSA sentiment analysis pipeline
        absa_model: ABSA model for label conversion
        text_col: Column name for text data (default: 'raw_text')
        aspects_col: Column name for aspect terms (default: 'predicted_aspects_with_cat')
    
    Returns:
        DataFrame with added 'aspects_with_sentiment' column
    """

    if isinstance(df[aspects_col].iloc[0], str):
        df['processed_aspects'] = df[aspects_col].progress_apply(json.loads)
    else:
        df['processed_aspects'] = df[aspects_col]
    
    # Use the pipeline version
    df['aspects_with_sentiment'] = df.apply(
        lambda row: predict_sentiment_for_aspects(
            text=row[text_col],
            aspects=row['processed_aspects'],
            absa_pipeline=absa_pipeline,
            absa_model=absa_model
        ),
        axis=1
    )
    
    return df

def similar(a, b):
    """
    Calculates similarity between two strings using SequenceMatcher.
    
    Args:
        a: First string to compare
        b: Second string to compare
    
    Returns:
        float: Similarity ratio (0.0 to 1.0)
    """
    return SequenceMatcher(None, a, b).ratio()    

def safe_load(data):
    """
    Safely parses string data into Python objects.
    
    Args:
        data: Input string or list to parse
    
    Returns:
        Parsed Python object (list/dict) or empty list if parsing fails
    """
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(data)
            except:
                return []
    elif isinstance(data, list):
        return data
    return []

def evaluate_sentiment_analysis(df, partial_threshold=0.7):
    """
    Evaluates sentiment analysis performance at aspect and category levels.
    
    Args:
        df: DataFrame containing ground truth and predicted aspects
        partial_threshold: Similarity threshold for partial matches (default: 0.7)
    
    Returns:
        Dictionary containing:
        - Evaluation metrics (precision/recall/f1) for exact/partial matches
        - Confusion matrices for different matching levels
        - Sentiment distribution by category
        - Combined evaluation results
    """
    
    results = {
        'aspect_level': {
            'exact': {'y_true': [], 'y_pred': []},
            'partial': {'y_true': [], 'y_pred': []}
        },
        'category_level': {
            'exact': {'y_true': [], 'y_pred': []},
            'partial': {'y_true': [], 'y_pred': []}
        },
        'category_sentiment_distribution': defaultdict(lambda: defaultdict(int)),
        'confusion_matrices': {
            'aspect_exact': defaultdict(lambda: defaultdict(int)),
            'aspect_partial': defaultdict(lambda: defaultdict(int)),
            'aspect_combined': defaultdict(lambda: defaultdict(int)),
            'category_exact': defaultdict(lambda: defaultdict(int)),
            'category_partial': defaultdict(lambda: defaultdict(int)),
            'category_combined': defaultdict(lambda: defaultdict(int))
        }
    }
    
    # Use German labels to match your data
    sentiment_labels = ['positiv', 'negativ', 'neutral']
    
    for _, row in df.iterrows():
        # Skip if no predicted aspects
        if not row['aspects_with_sentiment']:
            continue
            
        # Prepare true aspects (term, category, sentiment)
        true_aspects = []
        true_terms = safe_load(row['aspectTerms'])
        true_cats = safe_load(row['aspectCategories'])
            
        for i, term in enumerate(true_terms):
            if isinstance(term, dict) and term.get('term') == 'noAspectTerm':
                continue
                
            # Handle different term formats
            term_text = term['term'] if isinstance(term, dict) else term
            term_polarity = term['polarity'] if isinstance(term, dict) else 'neutral'
            
            # Get corresponding category
            cat = true_cats[i]['category'] if (i < len(true_cats) and isinstance(true_cats[i], dict)) else 'Allgemein'
            
            # Keep original German sentiment labels
            sentiment = term_polarity.lower()
            if sentiment not in sentiment_labels:
                sentiment = 'neutral'  # default fallback
                
            true_aspects.append({
                'term': term_text,
                'category': cat,
                'sentiment': sentiment
            })
        
        # Prepare predicted aspects - ensure they use German labels
        pred_aspects = []
        for aspect in row['aspects_with_sentiment']:
            # Convert English labels to German if needed
            pred_sentiment = aspect['sentiment'].lower()
            if pred_sentiment == 'positive':
                pred_sentiment = 'positiv'
            elif pred_sentiment == 'negative':
                pred_sentiment = 'negativ'
                
            pred_aspects.append({
                'term': aspect['term'],
                'category': aspect['category'],
                'sentiment': pred_sentiment
            })
        
        # ===== 1. Exact Aspect Term Matching =====
        matched_true = set()
        matched_pred = set()
        
        for i, pred in enumerate(pred_aspects):
            for j, true in enumerate(true_aspects):
                if pred['term'] == true['term']:
                    # Aspect-level evaluation
                    results['aspect_level']['exact']['y_true'].append(true['sentiment'])
                    results['aspect_level']['exact']['y_pred'].append(pred['sentiment'])
                    results['confusion_matrices']['aspect_exact'][true['sentiment']][pred['sentiment']] += 1
                    
                    # Category-level evaluation
                    if pred['category'] == true['category']:
                        results['category_level']['exact']['y_true'].append(true['sentiment'])
                        results['category_level']['exact']['y_pred'].append(pred['sentiment'])
                        results['confusion_matrices']['category_exact'][true['sentiment']][pred['sentiment']] += 1

                    # Combined-level evaluation
                    results['confusion_matrices']['aspect_combined'][true['sentiment']][pred['sentiment']] += 1
                    if pred['category'] == true['category']:
                        results['confusion_matrices']['category_combined'][true['sentiment']][pred['sentiment']] += 1

                    
                    matched_true.add(j)
                    matched_pred.add(i)
                    break
        
        # ===== 2. Partial Aspect Term Matching =====
        for i, pred in enumerate(pred_aspects):
            if i in matched_pred:
                continue
                
            for j, true in enumerate(true_aspects):
                if j in matched_true:
                    continue
                    
                if similar(pred['term'], true['term']) >= partial_threshold:
                    # Aspect-level evaluation
                    results['aspect_level']['partial']['y_true'].append(true['sentiment'])
                    results['aspect_level']['partial']['y_pred'].append(pred['sentiment'])
                    results['confusion_matrices']['aspect_partial'][true['sentiment']][pred['sentiment']] += 1
                    
                    # Category-level evaluation
                    if pred['category'] == true['category']:
                        results['category_level']['partial']['y_true'].append(true['sentiment'])
                        results['category_level']['partial']['y_pred'].append(pred['sentiment'])
                        results['confusion_matrices']['category_partial'][true['sentiment']][pred['sentiment']] += 1

                    # Combined-level evaluation
                    results['confusion_matrices']['aspect_combined'][true['sentiment']][pred['sentiment']] += 1
                    if pred['category'] == true['category']:
                        results['confusion_matrices']['category_combined'][true['sentiment']][pred['sentiment']] += 1

                    matched_true.add(j)
                    matched_pred.add(i)
                    break
        
        # ===== 3. Count sentiment distribution by category =====
        for pred in pred_aspects:
            results['category_sentiment_distribution'][pred['category']][pred['sentiment']] += 1
    
    # Calculate metrics for each evaluation type
    evaluation_types = [
        ('aspect_exact', results['aspect_level']['exact']),
        ('aspect_partial', results['aspect_level']['partial']),
        ('category_exact', results['category_level']['exact']),
        ('category_partial', results['category_level']['partial'])
    ]

    # ==== Combined aspect-level evaluation (exact + partial) ====
    combined_y_true = results['aspect_level']['exact']['y_true'] + results['aspect_level']['partial']['y_true']
    combined_y_pred = results['aspect_level']['exact']['y_pred'] + results['aspect_level']['partial']['y_pred']
    
    if combined_y_true:
        precision, recall, f1, _ = precision_recall_fscore_support(
            combined_y_true, combined_y_pred, labels=sentiment_labels, average=None, zero_division=0
        )
        results['aspect_combined'] = {
            'precision': dict(zip(sentiment_labels, precision)),
            'recall': dict(zip(sentiment_labels, recall)),
            'f1': dict(zip(sentiment_labels, f1))
        }
    else:
        results['aspect_combined'] = None

    # ==== Combined category-level evaluation (exact + partial) ====
    combined_cat_y_true = results['category_level']['exact']['y_true'] + results['category_level']['partial']['y_true']
    combined_cat_y_pred = results['category_level']['exact']['y_pred'] + results['category_level']['partial']['y_pred']
    
    if combined_cat_y_true:
        precision, recall, f1, _ = precision_recall_fscore_support(
            combined_cat_y_true, combined_cat_y_pred, labels=sentiment_labels, average=None, zero_division=0
        )
        results['category_combined'] = {
            'precision': dict(zip(sentiment_labels, precision)),
            'recall': dict(zip(sentiment_labels, recall)),
            'f1': dict(zip(sentiment_labels, f1))
        }
    else:
        results['category_combined'] = None

    
    for name, data in evaluation_types:
        if data['y_true']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                data['y_true'], data['y_pred'], labels=sentiment_labels, average=None, zero_division=0
            )
            results[name] = {
                'precision': dict(zip(sentiment_labels, precision)),
                'recall': dict(zip(sentiment_labels, recall)),
                'f1': dict(zip(sentiment_labels, f1))
            }
        else:
            results[name] = None
    
    return results

def prepare_absa_confusion_data(results, matrix_type):
    """
    Prepare y_true and y_pred arrays from ABSA results for a specific matrix type
    
    Args:
        results: ABSA evaluation results dictionary
        matrix_type: One of 'aspect_exact', 'aspect_partial', etc.
        
    Returns:
        Tuple of (y_true, y_pred) arrays
    """
    sentiment_labels = ['positiv', 'negativ', 'neutral']
    y_true = []
    y_pred = []
    
    if matrix_type not in results['confusion_matrices']:
        return [], []
    
    for true_label in sentiment_labels:
        for pred_label in sentiment_labels:
            count = results['confusion_matrices'][matrix_type][true_label].get(pred_label, 0)
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)
    
    return y_true, y_pred

def plot_absa_conf_matrix(y_true, y_pred, title):
    """
    Modified version of your plot_conf_matrix for ABSA sentiment analysis
    
    Args:
        y_true: Array of true sentiment labels (German)
        y_pred: Array of predicted sentiment labels (German)
        title: Plot title
    """
    sentiment_labels = ['positiv', 'negativ', 'neutral']
    
    cm = confusion_matrix(y_true, y_pred, labels=sentiment_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sentiment_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()

    #save
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
    save_path = f"./pipeline_results/conf_matrix/absa/conf_matrix_{filename}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nConfusion matrix saved to: {save_path}")
    
    plt.show()

def plot_all_absa_matrices(results):
    """
    Visualizes all ABSA confusion matrices from evaluation results.
    
    Args:
        results: Dictionary containing evaluation results from evaluate_sentiment_analysis()
    
    Outputs:
        Six confusion matrix plots showing performance at:
        - Aspect level (exact/partial/combined matches)
        - Category level (exact/partial/combined matches)
    """
    matrix_types = [
        ('aspect_exact', 'Aspect-Level (Exact Match)'),
        ('aspect_partial', 'Aspect-Level (Partial Match)'),
        ('category_exact', 'Category-Level (Exact Match)'),
        ('category_partial', 'Category-Level (Partial Match)'),
        ('aspect_combined', 'Aspect-Level (Exact + Partial)'),
        ('category_combined', 'Category-Level (Exact + Partial)')
    ]
    
    for matrix_type, title in matrix_types:
        y_true, y_pred = prepare_absa_confusion_data(results, matrix_type)
        if y_true:  # Only plot if there's data
            plot_absa_conf_matrix(y_true, y_pred, title)

def plot_sentiment_distribution_by_category(results, output_path):
    """
    Plot sentiment distribution for each category using the data from evaluation.
    
    Args:
        results: Evaluation results dictionary containing 'category_sentiment_distribution'
    """
    # Colors for each sentiment
    colors = {
        "positiv": "#8BC34A",  # Soft green
        "negativ": "#F94449",  # Soft red
        "neutral": "#607D8B"   # Soft blue-gray
    }

    # Get the sentiment distribution data
    sentiment_dist = results['category_sentiment_distribution']
    
    # Create individual plots for each category
    for category, values in sentiment_dist.items():
        plt.figure(figsize=(8, 6))
        
        # Prepare data for plotting
        labels = []
        sizes = []
        color_list = []
        n = sum(values.values())  # Total count for this category
        
        # Only include sentiments with count > 0
        sentiments_to_include = [s for s in ["positiv", "negativ", "neutral"] 
                               if values.get(s, 0) > 0]
        
        for sentiment in sentiments_to_include:
            count = values.get(sentiment, 0)
            percentage = (count / n) * 100
            labels.append(f"{percentage:.1f}%\n({count}/{n})") #{sentiment}\n
            sizes.append(percentage)
            color_list.append(colors[sentiment])
        
        # Create pie chart with labels inside the wedges
        wedges, texts = plt.pie(
            sizes,
            colors=color_list,
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            textprops={'fontsize': 10, 'ha': 'center', 'va': 'center'}
        )
        
        # Add labels inside the wedges
        for i, (wedge, label) in enumerate(zip(wedges, labels)):
            # Calculate position for the label
            angle = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))
            plt.text(x, y, label, ha='center', va='center', fontsize=12)
        
        # Add title with sample size
        plt.title(f"{category} (n={n})", fontsize=16, pad=5)

        # Add legend with sentiment labels
        plt.legend(
            wedges,
            sentiments_to_include,
            title="Sentiments",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        
        # Adjust layout to prevent cutting off labels
        plt.tight_layout()
        
        # Save each figure individually
        os.makedirs(output_path, exist_ok=True)  # Create the directory if it doesn't exist
        filename = f"sentiment_{category.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_path, filename), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    print(f"Sentiment distribution pie charts saved successfully to {output_path}!")

def print_evaluation_results(results):
    """
    Prints and visualizes ABSA evaluation results.
    
    Args:
        results: Dictionary containing evaluation metrics from evaluate_sentiment_analysis()
    
    Outputs:
        - Formatted performance metrics (precision/recall/F1) for:
          * Aspect-level (exact/partial/combined)
          * Category-level (exact/partial/combined)
        - Sentiment distribution by category
        - Confusion matrices for all evaluation types
        - Visualizations of results
    """
    # 1. Print aspect-level evaluation
    print("=== Aspect-Level Sentiment Evaluation ===")
    for match_type in ['exact', 'partial']:
        print(f"\n{match_type.capitalize()} Matches:")
        if f'aspect_{match_type}' in results and results[f'aspect_{match_type}']:
            for sentiment in ['positiv', 'negativ', 'neutral']:
                print(f"  {sentiment.capitalize()}:")
                print(f"    Precision: {results[f'aspect_{match_type}']['precision'].get(sentiment, 0):.4f}")
                print(f"    Recall:    {results[f'aspect_{match_type}']['recall'].get(sentiment, 0):.4f}")
                print(f"    F1-score:  {results[f'aspect_{match_type}']['f1'].get(sentiment, 0):.4f}")
        else:
            print("  No matches found")
    
    # 2. Print category-level evaluation
    print("\n=== Category-Level Sentiment Evaluation ===")
    for match_type in ['exact', 'partial']:
        print(f"\n{match_type.capitalize()} Matches:")
        if f'category_{match_type}' in results and results[f'category_{match_type}']:
            for sentiment in ['positiv', 'negativ', 'neutral']:
                print(f"  {sentiment.capitalize()}:")
                print(f"    Precision: {results[f'category_{match_type}']['precision'].get(sentiment, 0):.4f}")
                print(f"    Recall:    {results[f'category_{match_type}']['recall'].get(sentiment, 0):.4f}")
                print(f"    F1-score:  {results[f'category_{match_type}']['f1'].get(sentiment, 0):.4f}")
        else:
            print("  No matches found")

    # 3. Print combined aspect-level results
    print("\n=== Combined Aspect-Level Sentiment Evaluation (Exact + Partial) ===")
    if results.get('aspect_combined'):
        for sentiment in ['positiv', 'negativ', 'neutral']:
            print(f"  {sentiment.capitalize()}:")
            print(f"    Precision: {results['aspect_combined']['precision'].get(sentiment, 0):.4f}")
            print(f"    Recall:    {results['aspect_combined']['recall'].get(sentiment, 0):.4f}")
            print(f"    F1-score:  {results['aspect_combined']['f1'].get(sentiment, 0):.4f}")
    else:
        print("  No combined matches found")

    # 4. Print combined category-level results
    print("\n=== Combined Category-Level Sentiment Evaluation (Exact + Partial) ===")
    if results.get('category_combined'):
        for sentiment in ['positiv', 'negativ', 'neutral']:
            print(f"  {sentiment.capitalize()}:")
            print(f"    Precision: {results['category_combined']['precision'].get(sentiment, 0):.4f}")
            print(f"    Recall:    {results['category_combined']['recall'].get(sentiment, 0):.4f}")
            print(f"    F1-score:  {results['category_combined']['f1'].get(sentiment, 0):.4f}")
    else:
        print("  No combined category-level matches found")

    # 5. Add overall ABSA F1 score (macro-average of combined aspect-level results)
    print("\n=== Overall ABSA Performance ===")
    
    if results.get('aspect_combined'):
        evaluation_types = [
            ('aspect_exact', 'Aspect-Level (Exact)'),
            ('aspect_partial', 'Aspect-Level (Partial)'),
            ('category_exact', 'Category-Level (Exact)'), 
            ('category_partial', 'Category-Level (Partial)'),
            ('aspect_combined', 'Aspect-Level (Combined)'),
            ('category_combined', 'Category-Level (Combined)')
        ]
        
        for eval_type, display_name in evaluation_types:
            if eval_type in results and results[eval_type]:
                # Get F1 scores and confusion matrix
                f1_scores = results[eval_type]['f1']
                cm = results['confusion_matrices'][eval_type]
                
                # Calculate supports (true counts per class)
                supports = {
                    sentiment: sum(cm.get(sentiment, {}).values())
                    for sentiment in ['positiv', 'negativ', 'neutral']
                }
                total = sum(supports.values())
                
                # Macro-average (simple mean)
                macro_f1 = sum(f1_scores.values()) / 3
                
                # Weighted-average (support-weighted)
                weighted_f1 = sum(
                    f1_scores[sentiment] * (supports[sentiment] / total)
                    for sentiment in f1_scores
                ) if total > 0 else 0
                
                print(f"\n{display_name}:")
                print(f"  Macro-F1:    {macro_f1:.4f}")
                print(f"  Weighted-F1: {weighted_f1:.4f} (n={total})")
    else:
        print("No aspect-level results available for overall F1 calculation")
    
    # 6. Print sentiment distribution by category
    print("\n=== Sentiment Distribution by Category ===")
    for category, counts in sorted(results['category_sentiment_distribution'].items()):
        total = sum(counts.values())
        print(f"\n{category} (n={total}):")
        for sentiment in ['positiv', 'negativ', 'neutral']:
            count = counts.get(sentiment, 0)
            print(f"  {sentiment.capitalize()}: {count} ({count/total:.1%})")

    # 7. Add visualization of sentiment distribution by category
    plot_sentiment_distribution_by_category(results, "./pipeline_results/sentiment_by_cat/")
    
    # 8. Print confusion matrices
    print("\n=== Confusion Matrices ===")
    for matrix_type in ['aspect_exact', 'aspect_partial', 'category_exact', 'category_partial', 'aspect_combined', 'category_combined']:
        if results['confusion_matrices'][matrix_type]:
            print(f"\n{matrix_type.replace('_', ' ').title()}:")
            print("True \\ Predicted".ljust(15), end="")
            for pred in ['positiv', 'negativ', 'neutral']:
                print(pred[:5].ljust(10), end="")
            print()
            
            for true in ['positiv', 'negativ', 'neutral']:
                print(true[:12].ljust(15), end="")
                for pred in ['positiv', 'negativ', 'neutral']:
                    print(str(results['confusion_matrices'][matrix_type][true].get(pred, 0)).ljust(10), end="")
                print()

    # Add visualization at the end
    plot_all_absa_matrices(results)

def evaluate_pipeline_performance(cat_results, sentiment_results):
    """
    Computes ATE+ABSA pipeline metrics for exact/partial/combined matches.
    
    Args:
        cat_results (dict): ATE metrics from evaluate_with_partial_matches_cat()
            - Keys: 'exact', 'partial' → {'term', 'combined' metrics}
        sentiment_results (dict): ABSA metrics from evaluate_sentiment_analysis()
            - Keys: 'aspect_exact', 'aspect_partial', 'aspect_combined'

    Returns:
        dict: {
            'exact_terms': {'ate', 'absa', 'joint' metrics},  # Term-only
            'exact_full': {'ate', 'absa', 'joint' metrics},   # Term+category
            'partial_terms': {...},
            'partial_full': {...},
            'combined_terms': {...},
            'combined_full': {...}
        }
        All sub-dicts contain:
            - ate: {'f1', 'precision', 'recall'}
            - absa: {'macro_f1', 'avg_precision', 'avg_recall'}
            - joint: {'f1', 'combined_score'}
    """
    results = {
        'exact_terms': {},
        'exact_full': {},
        'partial_terms': {},
        'partial_full': {},
        'combined_terms': {},
        'combined_full': {}
    }
    
    # --- Exact Matches ---
    # Term-only (ATE)
    results['exact_terms']['ate'] = {
        'f1': cat_results['exact']['term']['f1'],
        'precision': cat_results['exact']['term']['precision'],
        'recall': cat_results['exact']['term']['recall']
    }
    
    # Term+Category (ATE)
    results['exact_full']['ate'] = {
        'f1': cat_results['exact']['combined']['f1'],
        'precision': cat_results['exact']['combined']['precision'],
        'recall': cat_results['exact']['combined']['recall']
    }
    
    # ABSA for exact Term matches
    absa_metrics = sentiment_results.get('aspect_exact', {})
    results['exact_terms']['absa'] = {
        'macro_f1': sum(absa_metrics['f1'].values())/3 if absa_metrics else 0,
        'avg_precision': sum(absa_metrics['precision'].values())/3 if absa_metrics else 0,
        'avg_recall': sum(absa_metrics['recall'].values())/3 if absa_metrics else 0
    }
    # ABSA for exact Category matches
    absa_metrics = sentiment_results.get('category_exact', {})
    results['exact_full']['absa'] = {
        'macro_f1': sum(absa_metrics['f1'].values())/3 if absa_metrics else 0,
        'avg_precision': sum(absa_metrics['precision'].values())/3 if absa_metrics else 0,
        'avg_recall': sum(absa_metrics['recall'].values())/3 if absa_metrics else 0
    }
    
    # Joint metrics
    results['exact_terms']['joint'] = {
        'f1': 2 * (results['exact_terms']['ate']['f1'] * results['exact_terms']['absa']['macro_f1']) / 
              (results['exact_terms']['ate']['f1'] + results['exact_terms']['absa']['macro_f1']) 
              if (results['exact_terms']['ate']['f1'] + results['exact_terms']['absa']['macro_f1']) > 0 else 0,
        'combined_score': (results['exact_terms']['ate']['f1'] + 
                          results['exact_terms']['absa']['macro_f1'] + 
                          (2 * (results['exact_terms']['ate']['f1'] * results['exact_terms']['absa']['macro_f1']) / 
                          (results['exact_terms']['ate']['f1'] + results['exact_terms']['absa']['macro_f1']))) / 3
    }
    
    results['exact_full']['joint'] = {
        'f1': 2 * (results['exact_full']['ate']['f1'] * results['exact_full']['absa']['macro_f1']) / 
              (results['exact_full']['ate']['f1'] + results['exact_full']['absa']['macro_f1']) 
              if (results['exact_full']['ate']['f1'] + results['exact_full']['absa']['macro_f1']) > 0 else 0,
        'combined_score': (results['exact_full']['ate']['f1'] + 
                          results['exact_full']['absa']['macro_f1'] + 
                          (2 * (results['exact_full']['ate']['f1'] * results['exact_full']['absa']['macro_f1']) / 
                          (results['exact_full']['ate']['f1'] + results['exact_full']['absa']['macro_f1']))) / 3
    }
    
    # --- Partial Matches ---
    # Term-only (ATE)
    results['partial_terms']['ate'] = {
        'f1': cat_results['partial']['term']['f1'],
        'precision': cat_results['partial']['term']['precision'],
        'recall': cat_results['partial']['term']['recall']
    }
    
    # Term+Category (ATE)
    results['partial_full']['ate'] = {
        'f1': cat_results['partial']['combined']['f1'],
        'precision': cat_results['partial']['combined']['precision'],
        'recall': cat_results['partial']['combined']['recall']
    }
    
    # ABSA for partial matches
    absa_metrics = sentiment_results.get('aspect_partial', {})
    results['partial_terms']['absa'] = {
        'macro_f1': sum(absa_metrics['f1'].values())/3 if absa_metrics else 0,
        'avg_precision': sum(absa_metrics['precision'].values())/3 if absa_metrics else 0,
        'avg_recall': sum(absa_metrics['recall'].values())/3 if absa_metrics else 0
    }
    
    absa_metrics = sentiment_results.get('category_partial', {}) 
    results['partial_full']['absa'] = {
        'macro_f1': sum(absa_metrics['f1'].values())/3 if absa_metrics else 0,
        'avg_precision': sum(absa_metrics['precision'].values())/3 if absa_metrics else 0,
        'avg_recall': sum(absa_metrics['recall'].values())/3 if absa_metrics else 0
    }
    
    # Joint metrics
    results['partial_terms']['joint'] = {
        'f1': 2 * (results['partial_terms']['ate']['f1'] * results['partial_terms']['absa']['macro_f1']) / 
              (results['partial_terms']['ate']['f1'] + results['partial_terms']['absa']['macro_f1']) 
              if (results['partial_terms']['ate']['f1'] + results['partial_terms']['absa']['macro_f1']) > 0 else 0,
        'combined_score': (results['partial_terms']['ate']['f1'] + 
                          results['partial_terms']['absa']['macro_f1'] + 
                          (2 * (results['partial_terms']['ate']['f1'] * results['partial_terms']['absa']['macro_f1']) / 
                          (results['partial_terms']['ate']['f1'] + results['partial_terms']['absa']['macro_f1']))) / 3
    }
    
    results['partial_full']['joint'] = {
        'f1': 2 * (results['partial_full']['ate']['f1'] * results['partial_full']['absa']['macro_f1']) / 
              (results['partial_full']['ate']['f1'] + results['partial_full']['absa']['macro_f1']) 
              if (results['partial_full']['ate']['f1'] + results['partial_full']['absa']['macro_f1']) > 0 else 0,
        'combined_score': (results['partial_full']['ate']['f1'] + 
                          results['partial_full']['absa']['macro_f1'] + 
                          (2 * (results['partial_full']['ate']['f1'] * results['partial_full']['absa']['macro_f1']) / 
                          (results['partial_full']['ate']['f1'] + results['partial_full']['absa']['macro_f1']))) / 3
    }
    
    # --- Combined Matches ---
    # Term-only (ATE)
    results['combined_terms']['ate'] = {
        'f1': cat_results['partial']['term']['f1'],  # Partial includes exact
        'precision': cat_results['partial']['term']['precision'],
        'recall': cat_results['partial']['term']['recall']
    }
    
    # Term+Category (ATE)
    results['combined_full']['ate'] = {
        'f1': cat_results['partial']['combined']['f1'],
        'precision': cat_results['partial']['combined']['precision'],
        'recall': cat_results['partial']['combined']['recall']
    }
    
    # ABSA for combined matches
    absa_metrics = sentiment_results.get('aspect_combined', {})
    results['combined_terms']['absa'] = {
        'macro_f1': sum(absa_metrics['f1'].values())/3 if absa_metrics else 0,
        'avg_precision': sum(absa_metrics['precision'].values())/3 if absa_metrics else 0,
        'avg_recall': sum(absa_metrics['recall'].values())/3 if absa_metrics else 0
    }
    
    absa_metrics = sentiment_results.get('category_combined', {})
    results['combined_full']['absa'] = {
        'macro_f1': sum(absa_metrics['f1'].values())/3 if absa_metrics else 0,
        'avg_precision': sum(absa_metrics['precision'].values())/3 if absa_metrics else 0,
        'avg_recall': sum(absa_metrics['recall'].values())/3 if absa_metrics else 0
    }
    
    # Joint metrics
    results['combined_terms']['joint'] = {
        'f1': 2 * (results['combined_terms']['ate']['f1'] * results['combined_terms']['absa']['macro_f1']) / 
              (results['combined_terms']['ate']['f1'] + results['combined_terms']['absa']['macro_f1']) 
              if (results['combined_terms']['ate']['f1'] + results['combined_terms']['absa']['macro_f1']) > 0 else 0,
        'combined_score': (results['combined_terms']['ate']['f1'] + 
                          results['combined_terms']['absa']['macro_f1'] + 
                          (2 * (results['combined_terms']['ate']['f1'] * results['combined_terms']['absa']['macro_f1']) / 
                          (results['combined_terms']['ate']['f1'] + results['combined_terms']['absa']['macro_f1']))) / 3
    }
    
    results['combined_full']['joint'] = {
        'f1': 2 * (results['combined_full']['ate']['f1'] * results['combined_full']['absa']['macro_f1']) / 
              (results['combined_full']['ate']['f1'] + results['combined_full']['absa']['macro_f1']) 
              if (results['combined_full']['ate']['f1'] + results['combined_full']['absa']['macro_f1']) > 0 else 0,
        'combined_score': (results['combined_full']['ate']['f1'] + 
                          results['combined_full']['absa']['macro_f1'] + 
                          (2 * (results['combined_full']['ate']['f1'] * results['combined_full']['absa']['macro_f1']) / 
                          (results['combined_full']['ate']['f1'] + results['combined_full']['absa']['macro_f1']))) / 3
    }
    
    return results


def print_pipeline_results(results):
    """
    Prints formatted evaluation metrics for all pipeline combinations.
    
    Args:
        results (dict): Output from evaluate_pipeline_performance() containing:
            - Exact/partial/combined matches
            - Term-only and term+category variants
            - ATE, ABSA, and Joint metrics for each
    
    Prints:
        Organized sections showing:
        1. Aspect Term Extraction (F1/Precision/Recall)
        2. Sentiment Analysis (Macro-F1/Avg Precision/Avg Recall) 
        3. Joint Pipeline Performance (Joint F1/Combined Score)
        For up to all 6 match types.
    """
    headers = [
        ("exact_terms", "Exact Matched Terms"),
        ("exact_full", "Exact Matched Terms + Category"),
        #("partial_terms", "Partial Matched Terms"),
        #("partial_full", "Partial Matched Terms + Category"),
        ("combined_terms", "Combined (Exact+Partial) Terms"),
        ("combined_full", "Combined (Exact+Partial) Terms + Category")
    ]
    
    for key, title in headers:
        print(f"\n=== {title.upper()} ===")
        print("\nAspect Term Extraction:")
        print(f"  F1: {results[key]['ate']['f1']:.4f}")
        print(f"  Precision: {results[key]['ate']['precision']:.4f}")
        print(f"  Recall: {results[key]['ate']['recall']:.4f}")
        
        print("\nSentiment Analysis (ABSA):")
        print(f"  Macro-F1: {results[key]['absa']['macro_f1']:.4f}")
        print(f"  Avg Precision: {results[key]['absa']['avg_precision']:.4f}")
        print(f"  Avg Recall: {results[key]['absa']['avg_recall']:.4f}")
        
        print("\nJoint Pipeline Performance:")
        print(f"  Joint F1: {results[key]['joint']['f1']:.4f}")
        print(f"  Combined Score: {results[key]['joint']['combined_score']:.4f}")

def merge_tokens_and_map_indices(tokens):
    """
    Merges wordpiece tokens and maps which original token indices belong to each word.
    
    Returns:
        words: List of merged words
        index_map: List of lists, each sublist contains indices of original tokens that formed one word
    """
    words = []
    index_map = []
    current_word = ""
    current_indices = []

    for i, token in enumerate(tokens):
        if token.startswith("##"):
            current_word += token[2:]
            current_indices.append(i)
        else:
            if current_word:
                words.append(current_word)
                index_map.append(current_indices)
            current_word = token
            current_indices = [i]

    if current_word:
        words.append(current_word)
        index_map.append(current_indices)

    return words, index_map

def aggregate_attention_by_word(attention, index_map):
    """
    Averages attention matrix over subword token indices to get word-level attention.
    
    Args:
        attention: [seq_len, seq_len] attention matrix
        index_map: Mapping from words to their original subword token indices
    
    Returns:
        word-level attention matrix
    """
    num_words = len(index_map)
    new_attention = np.zeros((num_words, num_words))

    for i, row_indices in enumerate(index_map):
        for j, col_indices in enumerate(index_map):
            sub_matrix = attention[np.ix_(row_indices, col_indices)]
            new_attention[i, j] = sub_matrix.mean()

    return new_attention


def plot_ate_attention_heatmap(text, model, tokenizer, title="ATE Attention Heatmap"):
    """
    Generates a heatmap showing which tokens the ATE model focuses on when predicting aspect terms.
    
    Args:
        text: Input sentence
        model: ATE model (BERT-based)
        tokenizer: ATE tokenizer
        title: Plot title
    """
    # Ensure model is in eval mode and get its device
    model.eval()
    device = next(model.parameters()).device  # Get model's device (e.g., 'cuda:0' or 'cpu')
    
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True).to(device)
    # Suppress the attention implementation warning
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Average attention across all layers and heads
    attentions = torch.mean(torch.stack(outputs.attentions), dim=(0, 1))  # [layers, heads, seq_len, seq_len]
    attention = torch.mean(attentions, dim=0).cpu()  # Average over positions
    
    # Prepare tokens for display (skip special tokens like [CLS])
    # Get original tokens and trim [CLS]/[SEP]
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
    attention = attention[1:-1, 1:-1] # Remove attention for special tokens
    
    # Merge tokens and map indices
    words, index_map = merge_tokens_and_map_indices(raw_tokens)
    
    # Aggregate attention
    attention = aggregate_attention_by_word(attention.numpy(), index_map)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention, #.numpy(), #attention.detach().numpy(),
        xticklabels=words,
        yticklabels=words,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f"
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    os.makedirs("./pipeline_results/heatmaps/ate/", exist_ok=True)
    filename = f"ate_heatmap_{hash(text)}.png"
    plt.savefig(f"./pipeline_results/heatmaps/ate/{filename}", bbox_inches='tight', dpi=300)
    plt.close()
    return filename

def plot_absa_sentiment_heatmap(text, aspect, model, tokenizer, title="ABSA Sentiment Heatmap"):
    """
    Generates a heatmap showing which tokens influence sentiment prediction for a given aspect.
    
    Args:
        text: Full sentence
        aspect: Target aspect term
        model: ABSA model
        tokenizer: ABSA tokenizer
        title: Plot title
    """
    # Ensure model is in eval mode and get its device
    model.eval()
    device = next(model.parameters()).device  # Get model's device (e.g., 'cuda:0' or 'cpu')
    
    # Format input as "[CLS] text Aspect: aspect [SEP]"
    input_text = f"{text} [SEP] {aspect}" 
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(device)
    
    # Suppress the attention implementation warning
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True) 
    
    # Average attention for sentiment prediction
    attentions = torch.mean(torch.stack(outputs.attentions), dim=(0, 1))  # [layers, heads, seq_len, seq_len]
    attention = torch.mean(attentions, dim=0).cpu()  # Average over positions
    
    # Get tokens (skip [CLS] and [SEP])
    # Get original tokens and trim [CLS]/[SEP]
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
    attention = attention[1:-1, 1:-1] # Remove attention for special tokens
    
    # Merge tokens and map indices
    words, index_map = merge_tokens_and_map_indices(raw_tokens)
    
    # Aggregate attention
    attention = aggregate_attention_by_word(attention.numpy(), index_map)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention,
        xticklabels=words,
        yticklabels=words,
        cmap="RdYlBu",
        center=0,  # Neutral sentiment at center
        annot=False
    )
    plt.title(f"{title}\nAspect: '{aspect}'", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    os.makedirs("./pipeline_results/heatmaps/absa/", exist_ok=True)
    filename = f"absa_heatmap_{hash(text + aspect)}.png"
    plt.savefig(f"./pipeline_results/heatmaps/absa/{filename}", bbox_inches='tight', dpi=300)
    plt.close()
    return filename

def analyze_general_sentiment(text, sentiment_pipeline):
    """
    Analyzes overall sentiment of text when no specific aspects are detected.
    
    Args:
        text: Input text to analyze (string)
        sentiment_pipeline: Initialized sentiment analysis pipeline
    
    Returns:
        Dictionary containing:
        - term: Always 'noAspectTerm'
        - category: Always 'Allgemein' (general)
        - sentiment: 'positiv', 'negativ', or 'neutral'
        - confidence: Prediction score (0.0-1.0)
    
    Handles English-to-German label conversion and provides fallback neutral sentiment on errors.
    """
    try:
        result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
        # Convert model output to your standard format
        sentiment_label = result['label'].lower()
        if sentiment_label == 'positive':
            sentiment_label = 'positiv'
        elif sentiment_label == 'negative':
            sentiment_label = 'negativ'
        return {
            'term': 'noAspectTerm',
            'category': 'Allgemein',
            'sentiment': sentiment_label,
            'confidence': result['score']
        }
    except Exception as e:
        print(f"Error analyzing sentiment for text: {text[:50]}... Error: {str(e)}")
        return {
            'term': 'noAspectTerm',
            'category': 'Allgemein',
            'sentiment': 'neutral',
            'confidence': 0.0
        }
    
def save_detailed_results(unlabeled_data, output_path):
    """
    Save detailed human-readable results alongside the CSV file
    
    Args:
        unlabeled_data: DataFrame containing the predictions
        output_path: Base directory for saving files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Text results file path
    text_result_file = os.path.join(output_path, "detailed_pipeline_results.txt")
    
    with open(text_result_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("="*60 + "\n")
        f.write("ASPECT-BASED SENTIMENT ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        # Process each row
        for idx, row in unlabeled_data.iterrows():
            # Get the text and predictions
            text = row['raw_text']
            
            # Handle both string and list formats for aspects
            if isinstance(row['aspects_with_sentiment'], str):
                aspects = json.loads(row['aspects_with_sentiment'])
            else:
                aspects = row['aspects_with_sentiment']
            
            # Write sentence
            f.write(f"Sentence {idx+1}:\n{text}\n\n")
            
            # Write aspects and sentiments
            if aspects and isinstance(aspects, list):
                f.write("Detected Aspects:\n")
                for aspect in aspects:
                    f.write(f"- {aspect['term']} ({aspect['category']}): {aspect['sentiment']} (confidence: {aspect.get('confidence', 1.0):.2f})\n")
            else:
                f.write("No aspects detected\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"Detailed text results saved to {text_result_file}")

# --- FINAL PIPELINE ---

def ate_absa(test_data, ate_model, ate_tokenizer, absa_model, absa_tokenizer):
    """
    Executes full ATE+ABSA pipeline on labeled data.
    
    Args:
        test_data: DataFrame with ground truth columns ('aspectTerms', 'aspectCategories')
        ate_model: Aspect Term Extraction model
        ate_tokenizer: Tokenizer for ATE model
        absa_model: ABSA sentiment model  
        absa_tokenizer: Tokenizer for ABSA model
    
    Returns:
        None (saves results to CSV and prints evaluations)
    Outputs:
        - Aspect term predictions with categories
        - Sentiment predictions for aspects
        - Evaluation metrics and visualizations
        - CSV file with all predictions
    """
    
    # Load spaCy for tokenization (same as during training)
    nlp = spacy.load("de_core_news_sm")

    #-----ATE CAT-----

    # Create a prediction pipeline for aspect terms with categories
    aspect_category_extractor = pipeline("token-classification", 
                                        model=ate_model, 
                                        tokenizer=ate_tokenizer, 
                                        aggregation_strategy="simple", # For grouping tokens 
                                        device=0 if torch.cuda.is_available() else -1)  # Use GPU if available

    print("\nPredict Aspect Terms and their Categories: ")
    # Predict for all sentences (with progress bar)
    tqdm.pandas()
    test_data['predicted_aspects_with_cat'] = test_data['raw_text'].progress_apply(lambda x: extract_aspects_with_categories(x, aspect_category_extractor))

    print("\nTest Results for EXACT and PARTIAL Aspect Terms Match and their Categories: ")
    # Run evaluation with partial matching
    cat_results = evaluate_with_partial_matches_cat(
        test_data,
        true_terms_col='aspectTerms',
        true_cat_col='aspectCategories',
        pred_col='predicted_aspects_with_cat',
        partial_threshold=0.7  # Adjust as needed
    )
    
    # Print comprehensive report
    print_cat_evaluation(test_data, cat_results)

    print("\nPlot all confusion matrices: exact match (Terms, Combines), Partial Match (Terms, Combined): ")
    # Plot all confusion matrices: exact match (Terms, Combines), Partial Match (Terms, Combined)
    plot_all_confusion_matrices_from_results(cat_results, threshold=0.7)


    #-----ABSA-----
    
    # Create an ABSA sentiment prediction pipeline
    absa_sentiment_pipeline = pipeline(
        "text-classification",  # Using text-classification for sequence classification
        model=absa_model,
        tokenizer=absa_tokenizer,
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
        function_to_apply="softmax"  # Apply softmax to get probabilities
    )

    print("\nTest Results for EXACT and PARTIAL Aspect Terms Match, their Categories for Sentiment Mapping: ")
    # 1. Add sentiment predictions
    test_data = add_sentiment_predictions(df=test_data, absa_pipeline=absa_sentiment_pipeline, absa_model=absa_model)
    # 2. Run the evaluation
    sentiment_results = evaluate_sentiment_analysis(test_data)
    # 3. Print the results
    print_evaluation_results(sentiment_results)

    # Evaluation of the full pipeline
    print(f"\n=== PIPELINE OVERALL PERFORMANCE ===")
    pipeline_results = evaluate_pipeline_performance(cat_results, sentiment_results)
    print_pipeline_results(pipeline_results)

    # Generate heatmaps for the first 5 examples
    num_examples = min(5, len(test_data))
    print(f"\nGenerating heatmaps for {num_examples} examples...")
    
    for idx in tqdm(range(num_examples), desc="Creating heatmaps"):
        text = test_data.iloc[idx]['raw_text']
        
        # ATE heatmap
        ate_heatmap_path = plot_ate_attention_heatmap(text, ate_model, ate_tokenizer)
        
        # ABSA heatmaps for each predicted aspect
        aspects_raw = test_data.iloc[idx]['aspects_with_sentiment']
        if isinstance(aspects_raw, str):
            try:
                aspects = json.loads(aspects_raw)
            except json.JSONDecodeError:
                aspects = ast.literal_eval(aspects_raw)  # as a fallback
        else:
            aspects = aspects_raw
        for aspect in aspects:
            if aspect['term'] != 'noAspectTerm':
                absa_heatmap_path = plot_absa_sentiment_heatmap(
                    text, 
                    aspect['term'], 
                    absa_model, 
                    absa_tokenizer
                )

    # Save results to CSV
    output_path = "./pipeline_results/test_data_after_pipeline.csv"
    # Drop the processed_aspects column
    test_data = test_data.drop(columns=['processed_aspects'])
    
    test_data.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")


def ate_absa_unlabeled(unlabeled_data, ate_model, ate_tokenizer, absa_model, absa_tokenizer):
    """
    Runs ATE+ABSA pipeline on unlabeled text data.
    
    Args:
        unlabeled_data: DataFrame containing only 'raw_text' column
        ate_model: Aspect Term Extraction model  
        ate_tokenizer: Tokenizer for ATE model
        absa_model: ABSA sentiment model
        absa_tokenizer: Tokenizer for ABSA model
    
    Returns:
        DataFrame enhanced with:
        - Predicted aspects and categories
        - Aspect sentiments
        - General sentiment for aspect-less text
    
    Outputs:
        - CSV file with predictions
        - Sentiment distribution visualizations
        - Detailed text results
    """
    # Load spaCy for tokenization
    nlp = spacy.load("de_core_news_sm")

    #-----ATE CAT-----
    # Create a prediction pipeline for aspect terms with categories
    aspect_category_extractor = pipeline(
        "token-classification", 
        model=ate_model, 
        tokenizer=ate_tokenizer, 
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )

    print("\nPredicting Aspect Terms and their Categories")
    tqdm.pandas()
    unlabeled_data['predicted_aspects_with_cat'] = unlabeled_data['raw_text'].progress_apply(
        lambda x: extract_aspects_with_categories(x, aspect_category_extractor)
    )

    #-----ABSA-----
    # Create an ABSA sentiment prediction pipeline
    absa_sentiment_pipeline = pipeline(
        "text-classification",
        model=absa_model,
        tokenizer=absa_tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        function_to_apply="softmax"
    )

    print("\nPredicting Sentiment for Extracted Aspects")
    # Add sentiment predictions
    unlabeled_data = add_sentiment_predictions(
        df=unlabeled_data,
        absa_pipeline=absa_sentiment_pipeline,
        absa_model=absa_model,
        aspects_col='predicted_aspects_with_cat'
    )

    # Initialize German sentiment analyzer for 'Allgemein'
    """german_sentiment_pipeline = pipeline(
        "text-classification",
        model="oliverguhr/german-sentiment-bert",
        tokenizer="oliverguhr/german-sentiment-bert",
        device=0 if torch.cuda.is_available() else -1
    )"""

    german_sentiment_pipeline = pipeline(
        "text-classification",
        model="aari1995/German_Sentiment",
        tokenizer="aari1995/German_Sentiment",
        device=0 if torch.cuda.is_available() else -1
    )

    print("\nAnalyzing general sentiment for sentences with no detected aspects")
    
    # Process each row with progress bar
    tqdm.pandas(desc="General sentiment analysis for 'Allgemein'")
    
    # First ensure aspects_with_sentiment is properly converted to list if it's a string
    if isinstance(unlabeled_data['aspects_with_sentiment'].iloc[0], str):
        unlabeled_data['aspects_with_sentiment'] = unlabeled_data['aspects_with_sentiment'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    
    # Analyze and update aspects_with_sentiment for empty cases
    unlabeled_data['aspects_with_sentiment'] = unlabeled_data.progress_apply(
        lambda row: (
            [analyze_general_sentiment(row['raw_text'], german_sentiment_pipeline)]
            if (isinstance(row['aspects_with_sentiment'], list) and 
                len(row['aspects_with_sentiment']) == 0)
            else row['aspects_with_sentiment']
        ),
        axis=1
    )

    #-----ANALYSIS AND VISUALIZATION-----
    
   # 1. Create sentiment distribution by category
    print("\nAnalyzing Sentiment Distribution by Category")
    category_sentiment_distribution = defaultdict(lambda: defaultdict(int))
    
    for _, row in tqdm(unlabeled_data.iterrows(), total=len(unlabeled_data), desc="Processing sentiment distribution"):
        # Handle both string and list formats for aspects
        if isinstance(row['aspects_with_sentiment'], str):
            aspects = json.loads(row['aspects_with_sentiment'])
        else:
            aspects = row['aspects_with_sentiment']
        
        if not isinstance(aspects, list):
            continue
            
        for aspect in aspects:
            if 'sentiment' in aspect:
                # Convert sentiment to lowercase for consistency
                sentiment = aspect['sentiment'].lower()
                if sentiment in ['positiv', 'negativ', 'neutral']:
                    category_sentiment_distribution[aspect['category']][sentiment] += 1
    
    # 2. Prepare the results structure expected by the plotting function
    results = {'category_sentiment_distribution': {k: dict(v) for k, v in category_sentiment_distribution.items()}}

    # Print sentiment distribution
    print("\n=== Sentiment Distribution by Category ===")
    for category, counts in sorted(results['category_sentiment_distribution'].items()):
        total = sum(counts.values())
        if total == 0:
            continue
        print(f"\n{category} (n={total}):")
        for sentiment in ['positiv', 'negativ', 'neutral']:
            count = counts.get(sentiment, 0)
            percentage = (count / total) if total > 0 else 0
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1%})")
    
    # 3. Plot sentiment distribution by category
    print("\nGenerating sentiment distribution plots")
    plot_sentiment_distribution_by_category(results, "./pipeline_results/unlabeled_sentiment_by_cat/")

    # 4. Generate heatmaps for a sample of the data
    sample_size = min(5, len(unlabeled_data))
    sample_data = unlabeled_data.sample(sample_size) if len(unlabeled_data) > 5 else unlabeled_data
    
    print(f"\nGenerating heatmaps for {sample_size} examples...")
    
    for idx, row in tqdm(sample_data.iterrows(), total=sample_size, desc="Creating heatmaps"):
        text = row['raw_text']
        
        # ATE heatmap
        ate_heatmap_path = plot_ate_attention_heatmap(text, ate_model, ate_tokenizer)
        
        # ABSA heatmaps for each aspect
        aspects_raw = row['aspects_with_sentiment']
        if isinstance(aspects_raw, str):
            try:
                aspects = json.loads(aspects_raw)
            except json.JSONDecodeError:
                aspects = ast.literal_eval(aspects_raw)  # as a fallback
        else:
            aspects = aspects_raw
        for aspect in aspects:
            if aspect['term'] != 'noAspectTerm':
                absa_heatmap_path = plot_absa_sentiment_heatmap(
                    text, 
                    aspect['term'], 
                    absa_model, 
                    absa_tokenizer
                )
    
    # 5. Save results
    output_path = "./pipeline_results/unlabeled_data_with_predictions.csv"
    
    # Convert JSON strings to lists for better readability
    if isinstance(unlabeled_data['predicted_aspects_with_cat'].iloc[0], str):
        unlabeled_data['predicted_aspects_with_cat'] = unlabeled_data['predicted_aspects_with_cat'].apply(json.loads)
    
    if isinstance(unlabeled_data['aspects_with_sentiment'].iloc[0], str):
        unlabeled_data['aspects_with_sentiment'] = unlabeled_data['aspects_with_sentiment'].apply(json.loads)

    # Drop the processed_aspects column
    unlabeled_data = unlabeled_data.drop(columns=['processed_aspects'])
    
    unlabeled_data.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    # Save detailed text results
    save_detailed_results(unlabeled_data, "./pipeline_results/")
    
    return unlabeled_data


    