from typing import List, Optional
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
import json

class TokenizerTrainer:
    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize the tokenizer trainer.
        
        Args:
            vocab_size: Size of the vocabulary
            min_frequency: Minimum frequency for a token to be included
            special_tokens: List of special tokens to add
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "<s>", "</s>", "<pad>", "<unk>", "<mask>"
        ]
        
    def _create_base_tokenizer(self) -> Tokenizer:
        """Create a base BPE tokenizer."""
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Add pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Add decoder
        tokenizer.decoder = decoders.ByteLevel()
        
        # Add post-processor
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        return tokenizer
    
    def _get_training_corpus(self, dataset_name: str, src_lang: str, tgt_lang: str):
        """
        Generator for the training corpus combining both languages.
        
        Args:
            dataset_name: Name of the dataset to use
            src_lang: Source language code
            tgt_lang: Target language code
            
        Yields:
            Batches of texts from both languages
        """
        # Create the configuration string (e.g., "de-en" for German-English)
        config = f"{tgt_lang}-{src_lang}" if f"{tgt_lang}-{src_lang}" in ["cs-en", "de-en", "fr-en", "hi-en", "ru-en"] else f"{src_lang}-{tgt_lang}"
        dataset = load_dataset(dataset_name, config)
        
        def batch_iterator():
            batch_size = 1000
            for i in range(0, len(dataset['train']), batch_size):
                batch = dataset['train'][i:i + batch_size]
                texts = []
                for item in batch['translation']:
                    texts.append(item[src_lang])
                    texts.append(item[tgt_lang])
                yield texts
        
        return batch_iterator()
    
    def train(
        self,
        dataset_name: str,
        src_lang: str,
        tgt_lang: str,
        output_dir: str,
        tokenizer_name: str
    ) -> PreTrainedTokenizerFast:
        """
        Train a single tokenizer on both source and target languages.
        
        Args:
            dataset_name: Name of the dataset to use
            src_lang: Source language code
            tgt_lang: Target language code
            output_dir: Directory to save the tokenizer
            tokenizer_name: Name for the tokenizer files
            
        Returns:
            Trained tokenizer wrapped in PreTrainedTokenizerFast
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        tokenizer = self._create_base_tokenizer()
        
        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        # Train tokenizer on combined languages
        tokenizer.train_from_iterator(
            self._get_training_corpus(dataset_name, src_lang, tgt_lang),
            trainer=trainer
        )
        
        # Save the tokenizer
        tokenizer_path = output_dir / f"{tokenizer_name}.json"
        tokenizer.save(str(tokenizer_path))
        
        # Save special tokens config
        special_tokens_path = output_dir / f"{tokenizer_name}_special_tokens.json"
        with open(special_tokens_path, 'w') as f:
            json.dump({
                'special_tokens': self.special_tokens,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang
            }, f, indent=2)
        
        # Create and return HuggingFace tokenizer
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )


if __name__ == "__main__":
    # Example usage
    trainer = TokenizerTrainer()
    
    # Train a single tokenizer for both languages
    tokenizer = trainer.train(
        dataset_name="wmt14",
        src_lang="en",  # English
        tgt_lang="de",  # German
        output_dir="tokenizers",
        tokenizer_name="en_de_tokenizer"
    )
    
    # Test tokenizer on both languages
    en_text = "Hello, this is a test sentence."
    de_text = "Hallo, das ist ein Testsatz."
    
    for text in [en_text, de_text]:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\nOriginal: {text}")
        print(f"Decoded: {decoded}") 