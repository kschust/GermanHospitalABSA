# GermanHospitalABSA

Repository for the Master's Thesis **"Aspect-Based Sentiment Analysis of German Hospital Reviews"**

## Motivation

Hospitals rely on patient feedback to improve healthcare services. Online reviews have become an important resource for understanding patient experience and provide valuable insights into healthcare services. However, analyzing unstructured reviews remains challenging, especially in low-resource languages like German. This thesis presents an Aspect-Based Sentiment Analysis (ABSA) approach for German hospital reviews.

## Repository Structure
- data/ # Data files
- functions/ # Utility functions
- pipeline_results/ # Pipeline output results

- 1-scraping_data.ipynb # Extraction of patient reviews from klinikbewertungen.de
- 2-data_exploration.ipynb # Exploratory Data Analysis (EDA)
- 3-data_preparation_for_labeling.ipynb # Data preparation for labeling
- 4-data_preprocessing.ipynb # Aspect Category Detection + preprocessing
- 5-data_labeling.ipynb # EDA of labeled data + format conversion
- 6-name_anonymization.ipynb # Data anonymization
- 7-ATE_OB.ipynb # ATE training (OB-Tagging)
- 7-ATE_OB_performance.ipynb # ATE performance analysis
- 7-ATE.ipynb # ATE training (OBI-Tagging)
- 8-ABSA.ipynb # ABSA training
- 8-ABSA_performance.ipynb # ABSA performance analysis
- 9-ATE_ABSA_pipeline.ipynb # Complete ATE+ABSA pipeline
- requirements.txt # Python dependencies


## Workflow Overview

1. **Data Collection & Preparation**
   - `1-scraping_data.ipynb`: Scrape patient reviews from klinikbewertungen.de
   - `2-data_exploration.ipynb`: Initial EDA
   - `3-data_preparation_for_labeling.ipynb`: Prepare data for manual labeling

2. **Labeling & Preprocessing**
   - `4-data_preprocessing.ipynb`: Aspect Category Detection + preprocessing
   - Manual labeling required between steps 4 and 5
   - `5-data_labeling.ipynb`: Process labeled data
   - `6-name_anonymization.ipynb`: Optional data anonymization

3. **Model Training**
   - `7-ATE_OB.ipynb`: Train Aspect Term Extraction (OB-Tagging)
   - `8-ABSA.ipynb`: Train Aspect-Based Sentiment Analysis
   - Performance analysis notebooks evaluate each component

4. **Pipeline Implementation**
   - `9-ATE_ABSA_pipeline.ipynb`: Combine best models into final pipeline

## Requirements

Python dependencies are listed in `requirements.txt`.

## Data Availability

The annotated German hospital review dataset, used in this work, can be requested from the author, as well as access to the best fine-tuned models and tokenizers.

## Computational Resources

Computations for this work (notebooks 6-9) were performed using resources of the Leipzig University Computing Center.
