
# PDF Question Answering System

This Python script implements a question answering system that processes PDF documents and answers questions based on their content. The system utilizes natural language processing techniques and pre-trained models to extract relevant information and generate answers.

## Overview

The system consists of several key components:

1. PDF text extraction
2. Text indexing and embedding
3. Semantic search for relevant context
4. Question answering based on retrieved context

## Input

The main function `main()` takes two inputs:

1. `pdf_path` (str): The file path to the PDF document to be processed.
2. `question` (str): The question to be answered based on the PDF content.

## Output

The system returns a string containing the answer to the given question based on the content of the PDF.

## Key Components

### PDF Text Extraction

The `pdf_to_text()` function uses the PyMuPDF library to convert a PDF file into plain text.

### Text Indexing and Embedding

The `build_index()` function creates a FAISS index from the extracted text using a sentence transformer model (default: 'all-MiniLM-L6-v2'). This index allows for efficient semantic search.

### Semantic Search

The `retrieve_relevant_text()` function performs a semantic search to find the most relevant sentences from the PDF content based on the input question.

### Question Answering

The `answer_question()` function uses a pre-trained question-answering model (default: 'roberta-base-squad2') to generate an answer based on the retrieved context.

## Usage

To use the system, ensure all required libraries are installed and the necessary pre-trained models are available. Then, you can use the `main()` function as follows:

```python
pdf_path = 'example.pdf'
question = 'What concerns did Trump raise about NATO during his campaign?'
answer = main(pdf_path, question)
print(f"Answer: {answer}")
```

## Dependencies

- PyMuPDF (fitz)
- FAISS
- Transformers
- SentenceTransformer
- PyTorch

## Notes

- The script assumes that the required pre-trained models ('all-MiniLM-L6-v2' and 'roberta-base-squad2') are available in the current working directory.
- The system's performance depends on the quality and relevance of the PDF content to the asked questions.
- For optimal results, ensure that the input question is clear and directly related to the content of the PDF.
