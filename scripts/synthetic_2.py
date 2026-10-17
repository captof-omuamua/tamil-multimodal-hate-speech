import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, RagRetriever, RagTokenForGeneration
import numpy as np
import os

# Ensure authentication for Hugging Face gated model (if required)
# Add your Hugging Face API token here if accessing a gated model
hf_token = "REDACTED_USE_HF_TOKEN_ENVIRONMENT_VARIABLE"

# Load Excel file containing video and text embeddings
file_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/combined_embeddings.xlsx"
output_file_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/synthetic.xlsx"

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load embeddings data
data = pd.read_excel(file_path)

# Exclude the first column (sample names)
sample_names = data.iloc[:, 0].values  # Store sample names if needed
data = data.iloc[:, 1:]  # Skip the first column
# Ensure data contains only numeric values and handle non-numeric data
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# Split video and text embeddings
video_embeddings = data.iloc[:, :768].values
text_embeddings = data.iloc[:, 768:].values

# Concatenate embeddings for RAG input
combined_embeddings = [torch.tensor(np.concatenate([video_embeddings[i], text_embeddings[i]])) 
                       for i in range(len(video_embeddings))]

# Load LLaMA model and tokenizer (if gated, pass token)
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

# Initialize RAG retriever and model
retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', use_auth_token=hf_token)
rag_model = RagTokenForGeneration.from_pretrained('facebook/rag-token-nq', retriever=retriever, use_auth_token=hf_token)

# Define function to generate captions with error handling
def generate_caption(embedding):
    try:
        input_tensor = embedding.unsqueeze(0)  # Add batch dimension
        input_ids = tokenizer("<|startoftext|>", return_tensors="pt").input_ids
        
        # RAG-based generation
        with torch.no_grad():
            output = rag_model.generate(input_ids=input_ids, context_input=input_tensor, max_length=50)
        
        # Decode generated tokens to string
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Error generating caption for embedding: {str(e)}")
        return "Error"

# Generate synthetic captions/comments for each embedding pair
captions = []
batch_size = 10  # Modify batch size as per your system's memory constraints

for i in range(0, len(combined_embeddings), batch_size):
    batch_embeddings = combined_embeddings[i:i + batch_size]
    
    # Process each embedding in the batch
    for embedding in batch_embeddings:
        caption = generate_caption(embedding)
        captions.append(caption)
        print(caption)  # Print each generated caption
    
    # Periodically save progress to prevent data loss
    output_df = pd.DataFrame({
        "Sample Name": sample_names[:len(captions)],  # Ensure names correspond to the valid embeddings
        "Generated Captions": captions
    })
    
    output_df.to_excel(output_file_path, index=False)
    print(f"Checkpoint saved at {output_file_path}")

# Final save of all generated captions
output_df = pd.DataFrame({
    "Sample Name": sample_names[:len(captions)],  # Ensure names correspond to the valid embeddings
    "Generated Captions": captions
})

output_df.to_excel(output_file_path, index=False)
print(f"Captions generation complete. Saved to {output_file_path}.")

