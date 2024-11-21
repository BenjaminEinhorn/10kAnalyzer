# 10kAnalyzer
Financial Document Processing with FinBERT and OpenAI
# Financial Filings Analysis Tool

## Overview
This repository contains a tool for analyzing financial filings (e.g., 10-K and 10-Q reports) using natural language processing (NLP). The tool supports extracting sections from filings, generating embeddings with FinBERT, retrieving similar clauses, and producing investment recommendations with OpenAI's GPT models.

## Features
- Extract and parse sections from financial filings.
- Generate embeddings for text using the FinBERT model.
- Find similar clauses using cosine similarity.
- Generate actionable investment recommendations.

## Requirements
The tool uses the following libraries and versions:
- `torch==2.2.2`
- `transformers==4.39.3`
- `scikit-learn==1.4.2`
- `openai==1.51.2`
- `numpy==1.26.4`
- `PyPDF2==3.0.1`
- `pdfminer.six==20240706`
