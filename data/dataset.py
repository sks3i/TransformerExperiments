from datasets import load_dataset
from torch.utils.data import DataLoader
from .collate import TranslationDataCollator
from transformers import PreTrainedTokenizer
from typing import Dict, Callable


def load_translation_dataset():
    # Load English-German translation dataset from WMT14
    translation_dataset = load_dataset("wmt14", "de-en")
    return translation_dataset

def load_summarization_dataset():
    # Load CNN/DailyMail dataset which is commonly used for summarization
    summarization_dataset = load_dataset("cnn_dailymail", "3.0.0")
    return summarization_dataset

def get_datasets():
    # Load both datasets
    translation_data = load_translation_dataset()
    summarization_data = load_summarization_dataset()
    
    return {
        "translation": translation_data,
        "summarization": summarization_data
    }

def get_dataset(dataset_name: str):
    if dataset_name == "translation":
        return load_translation_dataset()
    elif dataset_name == "summarization":
        return load_summarization_dataset()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

def get_data_loaders(
    dataset_name: str,
    batch_size: int,
    collate_fn: Callable
) -> Dict[str, DataLoader]:
    """
    Create data loaders with the translation collator.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for encoding texts
        batch_size: Batch size for the data loaders
        max_length: Maximum sequence length
        device: Device to place tensors on
        
    Returns:
        Dictionary containing train and validation data loaders
    """
    dataset = get_dataset(dataset_name)
    
    train_loader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return {
        "train": train_loader,
        "val": val_loader
    }


if __name__ == "__main__":
    # Test loading the datasets
    datasets = get_datasets()
    
    # Print sample from translation dataset
    print("\nTranslation Dataset Sample:")
    print(datasets["translation"]["train"][0])
    
    # Print sample from summarization dataset
    print("\nSummarization Dataset Sample:")
    print(datasets["summarization"]["train"][0])
