# Similarity Search using Intel® Extension for Scikit-learn and Haystack

This code demonstrates how to perform similarity search using Intel® Extension for Scikit-learn and Haystack. The script extracts information from a dataset, preprocesses it using Haystack, and then uses Intel's extension for optimized Scikit-learn to perform cosine similarity search.

## Requirements

- `sklearnex`: Intel® Extension for Scikit-learn
- `scikit-learn`: Scikit-learn library for machine learning
- `haystack`: Haystack library for document preprocessing
- `pandas`: Data manipulation library for sorting and displaying results

Install the required packages using the following:

```bash
pip install scikit-learn-intelex scikit-learn pandas farm-haystack[inference]

