from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from src.utils.constants import MODELS_DIR


def load_model(model_name):
    """
    Load the model and tokenizer from the Hugging Face Hub. If the model is not
    found, it will be downloaded and cached in the specified directory.

    Args:
        model_name (str): The name of the model to load.
    """
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODELS_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)

    return model, tokenizer


def get_embeddings(model, tokenizer, text):
    """
    Get the embeddings for a given text using the specified model and tokenizer.
    Args:
        text (str): The input text to encode.
        tokenizer: The tokenizer to use for encoding the text.
        model: The model to use for generating embeddings.

    Returns:
        torch.Tensor: The embeddings for the input text.
    """

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    model, tokenizer = load_model(model_name)
    text = "Hello, how are you?"
    embeddings = get_embeddings(model, tokenizer, text)
    print(embeddings)