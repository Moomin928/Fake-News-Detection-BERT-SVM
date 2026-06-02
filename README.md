# Machine Learning and Transformer-Based Models for Fake News Detection

This repository contains the official implementation and experimental logs for the thesis project: *Can a hybrid approach, combining BERT and SVM (BERT-SVM) outperform standalone BERT and SVM models?*

## Project Structure
* `fake_news_detection.py`: Standard Python script containing the full architecture pipeline (TF-IDF+SVM, end-to-end BERT, and Hybrid BERT-SVM).
* `experiment_notebook.ipynb`: Jupyter Notebook containing the full execution logs, accuracy/Macro-F1 scores, confusion matrices, and figures generated in Google Colab.

## Environment & Dependencies
The experiments were conducted using Google Colaboratory (T4 GPU).
* Python 3.12
* PyTorch 2.10
* Transformers 5.0
* Scikit-Learn
* LIME, Seaborn, Matplotlib

## How to Run
1. Clone the repository or download the files.
2. Ensure the text datasets (LIAR / FakeNewsNet) are properly loaded.
3. Execute the Python script:
```bash
   python fake_news_detection.py
