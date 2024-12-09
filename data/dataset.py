from datasets import load_dataset

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

if __name__ == "__main__":
    # Test loading the datasets
    datasets = get_datasets()
    
    # Print sample from translation dataset
    print("\nTranslation Dataset Sample:")
    print(datasets["translation"]["train"][0])
    
    # Print sample from summarization dataset
    print("\nSummarization Dataset Sample:")
    print(datasets["summarization"]["train"][0])
