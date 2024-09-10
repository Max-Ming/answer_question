import fitz  # PyMuPDF
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModel
from sentence_transformers import SentenceTransformer
import os
import torch
import torch.nn.functional as F

# 获得pdf文档
def pdf_to_text(pdf_path):
    """Converts a PDF file to plain text.
    
    Args:
        pdf_path (str): The file path to the PDF document to be converted.
    
    Returns:
        str: The extracted text content from all pages of the PDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def mean_pooling(model_output, attention_mask):
    """Performs mean pooling on token embeddings weighted by attention mask.
    
    Args:
        model_output (torch.Tensor): Output from a transformer model, typically containing token embeddings.
        attention_mask (torch.Tensor): Attention mask indicating which tokens are padding (0) and which are not (1).
    
    Returns:
        torch.Tensor: Mean-pooled representation of the input, where each token is weighted by the attention mask.
    """
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def build_index(text, model_name='./all-MiniLM-L6-v2'):
    """Builds a FAISS index from a given text using a specified language model.
    
    Args:
        text (str): The input text to be indexed.
        model_name (str): The name or path of the pre-trained model to use. Defaults to './all-MiniLM-L6-v2'.
    
    Returns:
        tuple: A tuple containing:
            - faiss.IndexFlatL2: The FAISS index built from the sentence embeddings.
            - list: A list of sentences extracted from the input text.
            - torch.nn.Module: The loaded language model.
            - transformers.PreTrainedTokenizer: The tokenizer associated with the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sentences = text.split('\n')
    embeddings = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        model_output = model(**embeddings)

    sentence_embeddings = mean_pooling(model_output, embeddings['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(sentence_embeddings.numpy())
    return index, sentences, model, tokenizer

def retrieve_relevant_text(question, index, sentences, model, tokenizer, k=30):
    """Retrieves relevant text based on a given question using semantic search.
    
    Args:
        question (str): The input question to search for relevant text.
        index (faiss.Index): A FAISS index containing sentence embeddings.
        sentences (list): A list of sentences corresponding to the indexed embeddings.
        model (torch.nn.Module): A pre-trained transformer model for generating embeddings.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer compatible with the model.
        k (int, optional): The number of top relevant sentences to retrieve. Defaults to 30.
    
    Returns:
        str: A concatenated string of the k most relevant sentences.
    """
    question_embedding = tokenizer([question], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**question_embedding)

    question_embedding = mean_pooling(model_output, question_embedding['attention_mask'])

    question_embedding = F.normalize(question_embedding, p=2, dim=1)
    distances, indices = index.search(question_embedding, k)
    # 获取相关句子
    relevant_texts = [sentences[idx] for idx in indices[0]]
    return ' '.join(relevant_texts)

def answer_question(question, context, model_name='./roberta-base-squad2'):
    """Answers a question based on the given context using a question-answering model.
    
    Args:
        question (str): The question to be answered.
        context (str): The context or passage containing the information to answer the question.
        model_name (str, optional): The path or name of the pre-trained model to use. Defaults to './roberta-base-squad2'.
    
    Returns:
        str: The answer to the question extracted from the context.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    result = nlp(question=question, context=context)
    return result['answer']

def main(pdf_path, question):
    """Main function to process a PDF and answer a question based on its content.
    
    Args:
        pdf_path (str): The file path to the PDF document to be processed.
        question (str): The question to be answered based on the PDF content.
    
    Returns:
        str: The answer to the given question based on the content of the PDF.
    """
    current_path = os.getcwd()
    all_MiniLM_L6_v2_path = os.path.join(current_path, 'all-MiniLM-L6-v2')
    roberta_base_squad2_path = os.path.join(current_path, 'roberta-base-squad2')
    text = pdf_to_text(pdf_path)
    index, sentences, model, tokenizer = build_index(text, model_name=all_MiniLM_L6_v2_path)
    context = retrieve_relevant_text(question, index, sentences, model, tokenizer)
    answer = answer_question(question, context, model_name=roberta_base_squad2_path)
    return answer

if __name__ == "__main__":
    pdf_path = 'example.pdf'  # 替换为你的PDF文件路径
    question = 'What concerns did Trump raise about NATO during his campaign?'  # 替换为你的问题
    answer = main(pdf_path, question)
    print(f"Answer: {answer}")