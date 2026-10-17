import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, RagRetriever, RagTokenForGeneration
import numpy as np

# Load Excel file containing video and text embeddings
file_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/combined_embeddings.xlsx"
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

# Load LLaMA model and tokenizer
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Update to the desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize RAG retriever and model
retriever = RagRetriever.from_pretrained('facebook/rag-token-nq')
rag_model = RagTokenForGeneration.from_pretrained('facebook/rag-token-nq', retriever=retriever)

# Define function to generate captions
def generate_caption(embedding):
    input_tensor = embedding.unsqueeze(0)  # Add batch dimension
    input_ids = tokenizer("<|startoftext|>", return_tensors="pt").input_ids
    
    # RAG-based generation
    with torch.no_grad():
        output = rag_model.generate(input_ids=input_ids, context_input=input_tensor, max_length=50)
    
    # Decode generated tokens to string
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate synthetic captions/comments for each embedding pair
captions = []
for embedding in combined_embeddings:
    caption = generate_caption(embedding)
    captions.append(caption)
    print(caption)  # Print each generated caption

# Optional: Save the generated captions back to an Excel file
output_df = pd.DataFrame({
    "Sample Name": sample_names[:len(captions)],  # Ensure names correspond to the valid embeddings
    "Generated Captions": captions
})
output_df.to_excel("/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/synthetic.xlsx", index=False)

print("Captions generation complete. Saved to 'synthetic_captions.xlsx'.")