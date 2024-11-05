# RAG Pipeline for Website Question Answering

A simple Retrieval-Augmented Generation (RAG) pipeline for answering questions based on website content. This project combines retrieval of relevant website information with generative models to deliver contextually accurate answers.

## Features

- **Website Content Parsing**: Extracts and indexes content from a specified website URL.
- **Retrieval-Augmented Generation (RAG)**: Uses a hybrid approach for question answering.
- **Streamlit Interface**: User-friendly web interface for inputting URLs and questions.

## Tools

- **Orchestration**: LangChain
- **Guardrails**: NeMo-Guardrails
- **Monitoring**: Langsmith
- **Retrieval**: Hugging Face
- **Vector Database**: Pinecone
- **Generation**: Groq, Llama3
- **Deployment**: Render

## Workflow Architecture

1. **Data Loading**: Accepts various data formats such as PDF, CSV, or URL links.
2. **Chunking**: Uses text splitters to divide content into manageable chunks.
3. **Embedding**: Generates embeddings from text using Hugging Face models.
4. **Vector Store**: Stores embeddings in Pinecone for efficient retrieval.
5. **User Query (Input)**: Receives the user's question.
6. **Input Check**: Uses Guardrails (NeMo-Guardrails) to ensure the question is within the scope of the dataset.
7. **Retriever**: Retrieves relevant chunks from Pinecone based on the user’s query.
8. **Generator**: Groq or Llama3 generates responses based on retrieved information.
9. **Monitoring**: Langsmith monitors responses to ensure output quality.
10. **Output**: The final answer is presented to the user.


## Getting Started

### Prerequisites

- **Python 3.9+**
  
### Run 

- Install dependencies:
  ```bash
  pip install -r requirements.txt

- Start the app
  ```bash
  streamlit run app.py

## Usage
1. Enter a Website URL in the sidebar.
2. Ask Questions based on the website content.
3. The RAG pipeline retrieves relevant information and generates answers.
