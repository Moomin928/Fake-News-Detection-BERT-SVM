# Machine Learning and Transformer-Based Models for Fake News Detection

This repository contains the official implementation and experimental deployment pipelines for the thesis project: *Can a hybrid approach, combining BERT and SVM (BERT-SVM) outperform standalone BERT and SVM models?*

## Project Components
* `fake_news_detection.py`: A comprehensive, production-ready standard Python script containing the whole computational pipeline—ranging from TF-IDF+SVM baseline tokenization to the end-to-end BERT classifier and the pipeline-based Hybrid BERT-SVM architecture.
* `experiment_notebook.ipynb`: A clean, lightweight Jupyter Notebook configuration file designed explicitly for direct execution in a Google Colaboratory environment. Outputs and execution states have been cleared beforehand to guarantee 100% stable rendering on GitHub's static web interface without markdown parsing errors.

## Environment & Dependencies
The experiments were successfully trained and evaluated using Google Colaboratory (T4 GPU node running on a free-tier quota limit).
* Python 3.12
* PyTorch 2.10
* Transformers 5.0
* Scikit-Learn
* LIME, Seaborn, Matplotlib

## How to Reproduce the Results
1. Clone the repository or download the source files.
2. Ensure that your local environment has the text datasets (LIAR / FakeNewsNet PolitiFact subset) correctly downloaded and referenced.
3. Open `experiment_notebook.ipynb` in Google Colab and click **"Run all"** to reproduce the training loops, evaluations, and confusion matrices natively. Alternatively, run the executable script directly:
   ```bash
   python fake_news_detection.py
