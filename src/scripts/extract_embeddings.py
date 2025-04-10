import os
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import h5py
import time
import json
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.constants import CHEMBL_DATA_FILE, CHEMBL_EMBEDDINGS_DIR, MODELS_DIR, LLAMA_3P3_70B_MODEL_DIR, LLAMA_3P3_70B_MODEL_NAME
from src.utils.tokens import HF_TOKEN

def load_model(model_name, cache_dir):
    """
    Load model with multi-GPU optimized settings
    Args:
        model_name (str): The name of the model to load.
        cache_dir (str): The directory to cache the model.

    Returns:
        model: The loaded model.
        tokenizer: The tokenizer for the model.
    """
    model_config = {
        "cache_dir": cache_dir,
        "device_map": "auto",          # Auto-split across available GPUs
        "low_cpu_mem_usage": True,     
        "torch_dtype": torch.bfloat16,   
        "trust_remote_code": True,      # Required for Llama models
        "token": HF_TOKEN
    }

    if os.path.exists(cache_dir):
        print(f"Using cached model in {cache_dir}")
        model = AutoModel.from_pretrained(cache_dir, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    else:
        print(f"Loading model from {model_name} in {cache_dir}")
        model = AutoModel.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(cache_dir)
        tokenizer.save_pretrained(cache_dir)

    # Configure tokenizer for padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer


def get_embeddings(model, tokenizer, texts):
    """
    Get the embeddings for a given list of strings using the specified model and tokenizer.
    Args:
        texts (list): A list of strings to encode.
        model: The model to use for generating embeddings.
        tokenizer: The tokenizer to use for encoding the text.

    Returns:
        torch.Tensor: The embeddings for the input text.
    """
    # Tokenize the input text
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Move the inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass to get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # Get the mean of the last hidden state across the sequence length (do not include padding)
    attention_mask = inputs["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    return embeddings


def get_chemBL_data(chemBL_file_path):
    """
    Get the abstracts from the ChemBL data file.
    """
    with h5py.File(chemBL_file_path, 'r') as f:
        paper_ids = list(f.keys())

        abstracts = []
        canon_SMILES_lists = []
        for paper_name in tqdm(paper_ids):
            h5_dataset = f[paper_name]
            abstract = h5_dataset['abstract'][()].decode('utf-8')
            abstracts.append(abstract)
            compounds_list = json.loads(h5_dataset['compounds'][()].decode('utf-8'))
            canon_SMILES = [compound['canonical_smiles'] for compound in compounds_list]
            canon_SMILES_lists.append(canon_SMILES)

    return abstracts, canon_SMILES_lists


def main():
    model_name = LLAMA_3P3_70B_MODEL_NAME
    cache_dir = LLAMA_3P3_70B_MODEL_DIR
    print("Loading model...")
    model, tokenizer = load_model(model_name, cache_dir)

    # Get the ChemBL data
    chemBL_file_path = CHEMBL_DATA_FILE
    print("Getting ChemBL data...")
    abstracts, canon_SMILES_lists = get_chemBL_data(chemBL_file_path)

    # Get the embeddings
    print("Getting embeddings...")
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(abstracts)//10, batch_size)): # test 20% of data
        batch = abstracts[i:i + batch_size]
        batch_embeddings = get_embeddings(model, tokenizer, batch)
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)

    # Save the embeddings and canon_SMILES_lists as hdf5 file
    print("Saving embeddings and canon_SMILES_lists...")
    os.makedirs(CHEMBL_EMBEDDINGS_DIR, exist_ok=True)
    embeddings_file_path = CHEMBL_EMBEDDINGS_DIR + f"{time.strftime('%Y%m%d_%H%M%S')}_LLAMA_3P3_70B_embeddings.h5"
    if os.path.exists(embeddings_file_path):
        print(f"File {embeddings_file_path} already exists. Exiting.")
        return
    
    with h5py.File(embeddings_file_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings.cpu().numpy())
        f.create_dataset('canon_SMILES_lists', data=canon_SMILES_lists)

    print("Done!")


if __name__ == "__main__":
    main()