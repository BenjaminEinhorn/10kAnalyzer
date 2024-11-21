import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import openai
import numpy as np
from PyPDF2 import PdfReader
import os
from pdfminer.high_level import extract_text

# Use OpenAI API client
client = openai

# Change directory to where the files are located
os.chdir(r'E:\Data\Dataset\Finbert')

# Load FinBERT tokenizer and model for embedding generation
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")


def pdf_to_text_with_pdfminer(pdf_filepath, output_filepath):
    """
    Convert a PDF file to a plain text file using pdfminer.six.
    Args:
        pdf_filepath: Path to the input PDF file.
        output_filepath: Path to the output text file.
    """
    text = extract_text(pdf_filepath)
    with open(output_filepath, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text successfully extracted to {output_filepath}")


# Convert a sample PDF to text
pdf_to_text_with_pdfminer("10kNvidia.pdf", "10k_filing.txt")


def load_text_file(filepath):
    """
    Load the content of a text file.
    Args:
        filepath: Path to the text file.
    Returns:
        The content of the file as a string.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
    return text


# Load and print the text file
file_path = "10k_filing.txt"
full_text = load_text_file(file_path)
print(f"Loaded text with {len(full_text)} characters.")


def parse_10q_sections(text):
    """
    Parse the text of a 10-Q filing into key sections based on predefined markers.
    Args:
        text: The full text of the filing.
    Returns:
        A dictionary with section names as keys and content as values.
    """
    sections = {}
    section_markers = {
        "Financial Statements": "Item 1. Financial Statements (Unaudited)",
        "MD&A": "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "Market Risk": "Item 3. Quantitative and Qualitative Disclosures About Market Risk",
        "Controls and Procedures": "Item 4. Controls and Procedures",
        "Legal Proceedings": "Item 1. Legal Proceedings",
        "Risk Factors": "Item 1A. Risk Factors",
    }

    for section, marker in section_markers.items():
        start = text.find(marker)
        if start != -1:
            next_markers = [text.find(next_marker, start) for next_marker in section_markers.values() if text.find(next_marker, start) > start]
            end = min(next_markers, default=len(text))
            sections[section] = text[start:end].strip()

    return sections


# Parse the 10-K text into sections
parsed_sections = parse_10q_sections(full_text)

# Create a database of clauses
clause_database = [content for content in parsed_sections.values() if content]


def get_finbert_embedding(text):
    """
    Generate an embedding for the given text using FinBERT.
    Args:
        text: Input text string.
    Returns:
        A tensor containing the embedding.
    """
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token embedding


# Precompute embeddings for clauses
clause_embeddings = torch.cat([get_finbert_embedding(clause) for clause in clause_database], dim=0)
print(f"Precomputed embeddings for {len(clause_database)} sections.")


def find_similar_clauses(query, clause_database, clause_embeddings, top_n=3):
    """
    Retrieve the most similar clauses to the query using cosine similarity.
    Args:
        query: The query string.
        clause_database: List of clauses in the database.
        clause_embeddings: Precomputed embeddings of the clauses.
        top_n: Number of top similar clauses to retrieve.
    Returns:
        A tuple of the similar clauses and their similarity scores.
    """
    query_embedding = get_finbert_embedding(query)
    similarities = cosine_similarity(query_embedding, clause_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    similar_clauses = [clause_database[i] for i in top_indices]
    similarity_scores = similarities[0][top_indices]
    return similar_clauses, similarity_scores


def split_text_into_chunks_with_overlap(text, max_tokens=2000, overlap_sentences=2):
    """
    Split text into chunks with overlapping sentences.
    Args:
        text: Input text to split.
        max_tokens: Maximum tokens per chunk (approximation: 4 characters per token).
        overlap_sentences: Number of overlapping sentences between chunks.
    Returns:
        A list of text chunks.
    """
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    overlap = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) // 4 > max_tokens:
            chunks.append(". ".join(overlap + current_chunk) + ".")
            overlap = current_chunk[-overlap_sentences:]
            current_chunk = []

    if current_chunk:
        chunks.append(". ".join(overlap + current_chunk) + ".")
    return chunks


def generate_investor_recommendations_chunked(query_clause, similar_clauses, openai_key, max_tokens=2000):
    """
    Generate recommendations using chunked context for OpenAI GPT.
    Args:
        query_clause: The primary clause for the query.
        similar_clauses: List of similar clauses to provide context.
        openai_key: OpenAI API key.
        max_tokens: Maximum tokens per chunk.
    Returns:
        A list of recommendations.
    """
    openai.api_key = openai_key
    context = "\n\n".join(f"Relevant Clause {i+1}: {clause}" for i, clause in enumerate(similar_clauses))
    context_chunks = split_text_into_chunks_with_overlap(context, max_tokens=max_tokens, overlap_sentences=2)
    
    recommendations = []
    for i, chunk in enumerate(context_chunks):
        print(f"Processing chunk {i + 1} of {len(context_chunks)}...")
        content = f"""
        You are an investment advisor analyzing a company's 10-K filing. Based on the following clause and related context,
        what actions do you recommend investors take and why?

        Primary Clause:
        {query_clause}

        Context from Similar Clauses (Chunk {i + 1}):
        {chunk}
        """
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": content}],
                model="gpt-4",
            )
            recommendations.append(response.choices[0].message.content)
        except Exception as e:
            recommendations.append(f"Error processing chunk {i + 1}: {str(e)}")
    return recommendations


def consolidate_recommendations(chunk_recommendations, query_clause, openai_key):
    """
    Consolidate chunk-level recommendations into a single response.
    Args:
        chunk_recommendations: List of recommendations for individual chunks.
        query_clause: The primary clause for the query.
        openai_key: OpenAI API key.
    Returns:
        The consolidated recommendation.
    """
    openai.api_key = openai_key
    consolidated_context = "\n\n".join(f"Chunk {i+1} Recommendation: {rec}" for i, rec in enumerate(chunk_recommendations))
    content = f"""
    Based on the following recommendations for different parts of the context, 
    provide a consolidated investor recommendation.

    Query Clause:
    {query_clause}

    Chunk-Level Recommendations:
    {consolidated_context}
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error consolidating recommendations: {str(e)}"


# Example Query
query_clause = "What aspects of the company's future are unclear?"
similar_clauses, scores = find_similar_clauses(query_clause, clause_database, clause_embeddings)

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    chunk_recommendations = generate_investor_recommendations_chunked(query_clause, similar_clauses, openai_key)
    consolidated_recommendation = consolidate_recommendations(chunk_recommendations, query_clause, openai_key)
    print("Final Consolidated Recommendation:")
    print(consolidated_recommendation)
else:
    print("Please set the OPENAI_API_KEY environment variable.")
