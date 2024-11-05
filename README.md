# RAG Pipeline for Website Question Answering

A simple Retrieval-Augmented Generation (RAG) pipeline for answering questions based on website content. This project combines retrieval of relevant website information with generative models to deliver contextually accurate answers.

## Features

- **Website Content Parsing**: Extracts and indexes content from a specified website URL.
- **Retrieval-Augmented Generation (RAG)**: Uses a hybrid approach for question answering.
- **Streamlit Interface**: User-friendly web interface for inputting URLs and questions.


## Getting Started

### Prerequisites

- **Python 3.9+**
- Install dependencies:
  ```bash
  pip install -r requirements.txt

## Run the app
> streamlit run app.py

### Usage
1. Enter a Website URL in the sidebar.
2. Ask Questions based on the website content.
3. The RAG pipeline retrieves relevant information and generates answers.