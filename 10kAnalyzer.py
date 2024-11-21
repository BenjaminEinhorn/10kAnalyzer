import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import openai
import numpy as np
from PyPDF2 import PdfReader
import os
from pdfminer.high_level import extract_text

client = openai
# Change directory to where the files are located
os.chdir(r'E:\Data\Dataset\Finbert')

# Load FinBERT tokenizer and model for embedding generation
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")


def pdf_to_text_with_pdfminer(pdf_filepath, output_filepath):
    """
    Convert a PDF file to a plain text file using pdfminer.six.
    """
    text = extract_text(pdf_filepath)

    # Save the text to a file
    with open(output_filepath, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Text successfully extracted to {output_filepath}")


# Example usage
pdf_to_text_with_pdfminer("10kNvidia.pdf", "10k_filing.txt")


def load_text_file(filepath):
    """
    Load the content of a text file.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
    return text


# Example usage
file_path = "10k_filing.txt"
full_text = load_text_file(file_path)
print(f"Loaded text with {len(full_text)} characters.")


def parse_10q_sections(text):
    """
    Parse the text of a 10-Q filing into sections based on NVIDIA's table of contents.
    """
    sections = {}
    # Define markers for key sections (adjusted for NVIDIA 10-Q)
    section_markers = {
        "Financial Statements": "Item 1. Financial Statements (Unaudited)",
        "MD&A": "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "Market Risk": "Item 3. Quantitative and Qualitative Disclosures About Market Risk",
        "Controls and Procedures": "Item 4. Controls and Procedures",
        "Legal Proceedings": "Item 1. Legal Proceedings",
        "Risk Factors": "Item 1A. Risk Factors",
    }

    # Extract sections based on markers
    for section, marker in section_markers.items():
        start = text.find(marker)
        if start != -1:
            # Find the start of the next section
            next_markers = [text.find(next_marker, start) for next_marker in section_markers.values() if text.find(next_marker, start) > start]
            end = min(next_markers, default=len(text))
            sections[section] = text[start:end].strip()

    return sections


# Parse the 10-K into sections
parsed_sections = parse_10q_sections(full_text)

# Populate clause_database with sections
clause_database = [content for content in parsed_sections.values() if content]

# Helper function to get embeddings for a clause using FinBERT
def get_finbert_embedding(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    return embedding

# Precompute embeddings for the clauses in the database using FinBERT
clause_embeddings = torch.cat([get_finbert_embedding(clause) for clause in clause_database], dim=0)
print(f"Precomputed embeddings for {len(clause_database)} sections.")


# Function to retrieve similar clauses
def find_similar_clauses(query, clause_database, clause_embeddings, top_n=3):
    query_embedding = get_finbert_embedding(query)
    similarities = cosine_similarity(query_embedding, clause_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    similar_clauses = [clause_database[i] for i in top_indices]
    similarity_scores = similarities[0][top_indices]
    return similar_clauses, similarity_scores


# Function to split text into manageable chunks
def split_text_into_chunks_with_overlap(text, max_tokens=2000, overlap_sentences=2):
    """
    Split a long text into chunks of approximately max_tokens with overlapping sentences between chunks.
    """
    sentences = text.split(". ")  # Split text into sentences
    chunks = []
    current_chunk = []
    overlap = []

    for sentence in sentences:
        current_chunk.append(sentence)
        # Check if current chunk exceeds the token limit (approximation: ~4 characters = 1 token)
        if len(" ".join(current_chunk)) // 4 > max_tokens:
            chunks.append(". ".join(overlap + current_chunk) + ".")  # Add chunk with overlap
            overlap = current_chunk[-overlap_sentences:]  # Save last few sentences for overlap
            current_chunk = []  # Reset for the next chunk

    # Add the remaining sentences as the last chunk
    if current_chunk:
        chunks.append(". ".join(overlap + current_chunk) + ".")

    return chunks


# Function to generate recommendations by chunking the context
def generate_investor_recommendations_chunked(query_clause, similar_clauses, openai_key, max_tokens=2000):
    openai.api_key = openai_key

    # Create the context by joining relevant clauses
    context = "\n\n".join(f"Relevant Clause {i+1}: {clause}" for i, clause in enumerate(similar_clauses))
    
    # Split the context into manageable chunks
    context_chunks = split_text_into_chunks_with_overlap(context, max_tokens=max_tokens, overlap_sentences=2)
    
    recommendations = []
    for i, chunk in enumerate(context_chunks):
        print(f"Processing chunk {i + 1} of {len(context_chunks)}...")
        # Construct the prompt with the query clause and the current chunk
        content = f"""
        You are an investment advisor analyzing a company's 10-K filing. Based on the following clause and related context,
        what actions do you recommend investors take and why?

        Primary Clause:
        {query_clause}

        Context from Similar Clauses (Chunk {i + 1}):
        {chunk}
        """
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="gpt-4"
    )
            recommendations.append(response.choices[0].message.content)
        except Exception as e:
            recommendations.append(f"Error processing chunk {i + 1}: {str(e)}")

    # Combine recommendations from all chunks
    return recommendations


# Function to consolidate chunk-level recommendations
def consolidate_recommendations(chunk_recommendations, query_clause, openai_key):
    openai.api_key = openai_key
    
    # Combine all chunk recommendations
    consolidated_context = "\n\n".join(f"Chunk {i+1} Recommendation: {rec}" for i, rec in enumerate(chunk_recommendations))

    # Final prompt for consolidation
    content = f"""
    Based on the following recommendations for different parts of the context, 
    provide a consolidated investor recommendation.

    Query Clause:
    {query_clause}

    Chunk-Level Recommendations:
    {consolidated_context}
    """
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,  # Use the constructed content
                }
            ],
            model="gpt-4",  # Adjust model if needed
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error consolidating recommendations: {str(e)}"


# Example Query
query_clause = "What aspects of the companies future are unclear?"
similar_clauses, scores = find_similar_clauses(query_clause, clause_database, clause_embeddings)

# Ensure OpenAI API key is set
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    # Step 1: Generate chunked recommendations
    chunk_recommendations = generate_investor_recommendations_chunked(query_clause, similar_clauses, openai_key)
    
    # Step 2: Consolidate recommendations
    consolidated_recommendation = consolidate_recommendations(chunk_recommendations, query_clause, openai_key)
    
    # Print the final consolidated recommendation
    print("Final Consolidated Recommendation:")
    print(consolidated_recommendation)
else:
    print("Please set the OPENAI_API_KEY environment variable.")
