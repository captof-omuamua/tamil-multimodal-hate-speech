import os
import pandas as pd
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Specify directory paths
docx_directory = 'path_to_your_docx_files'  # Replace with your .docx files directory
output_excel_path = 'synthetic_tamil_transcripts.xlsx'  # Output Excel file

# Load BLOOM-560M model and tokenizer
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize lists to store data
synthetic_data = []

# Function to generate synthetic transcripts
def generate_synthetic_transcript(original_text, num_synth=100):
    synthetic_texts = []
    for i in range(1, num_synth + 1):
        # Prepare prompt with slight instruction to create variation
        prompt = f"Original Tamil text:\n{original_text}\n\nGenerate a similar but slightly different Tamil text."
        
        # Tokenize input and generate synthetic text
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,  # Adjust if needed based on text length
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode output and append
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        synthetic_texts.append(generated_text)
    return synthetic_texts

# Process each .docx file in the directory
for filename in sorted(os.listdir(docx_directory)):
    if filename.endswith('.docx') and filename.startswith('TAM_'):
        # Get file number Y
        file_number = filename.split('_')[1].split('.')[0]
        
        # Load the .docx file and extract text
        file_path = os.path.join(docx_directory, filename)
        doc = Document(file_path)
        original_text = '\n'.join([para.text for para in doc.paragraphs])
        
        # Generate 100 synthetic transcripts for this document
        synthetic_transcripts = generate_synthetic_transcript(original_text, num_synth=100)
        
        # Append data to list with naming convention
        for i, synthetic_text in enumerate(synthetic_transcripts, start=1):
            synthetic_data.append({
                "Synthetic_ID": f"{str(i).zfill(2)}_TAM_{file_number}",
                "Synthetic_Transcript": synthetic_text
            })

# Create a DataFrame and save to Excel
df = pd.DataFrame(synthetic_data)
df.to_excel(output_excel_path, index=False, engine='openpyxl')

print("Synthetic transcripts generated and saved to Excel.")
