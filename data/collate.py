from abc import ABC, abstractmethod
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizer


class BaseDataCollator(ABC):
    """Abstract base class for data collators."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        device: str = 'cuda',
        pad_to_multiple_of: Optional[int] = None
    ):
        """
        Initialize the base collator.
        
        Args:
            tokenizer: Tokenizer to use for encoding texts
            max_length: Maximum sequence length
            device: Device to place tensors on
            pad_to_multiple_of: Pad sequence length to be multiple of this value
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.pad_to_multiple_of = pad_to_multiple_of

    def _tokenize_and_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary containing input_ids, attention_mask, and other relevant tensors
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }

    @abstractmethod
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of examples.
        
        Args:
            batch: List of dictionaries containing data
            
        Returns:
            Dictionary containing processed tensors
        """
        pass

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True
        )


class TranslationDataCollator(BaseDataCollator):
    """Collator specific to translation tasks."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        src_lang: str = 'en',
        tgt_lang: str = 'de',
        max_length: int = 128,
        device: str = 'cuda',
        pad_to_multiple_of: Optional[int] = None
    ):
        """
        Initialize the translation collator.
        
        Args:
            tokenizer: Tokenizer to use for encoding texts
            src_lang: Source language code
            tgt_lang: Target language code
            max_length: Maximum sequence length
            device: Device to place tensors on
            pad_to_multiple_of: Pad sequence length to be multiple of this value
        """
        super().__init__(tokenizer, max_length, device, pad_to_multiple_of)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Process a batch of translation examples.
        
        Args:
            batch: List of dictionaries containing translation pairs
            
        Returns:
            Dictionary containing processed tensors for source and target
        """
        # Extract source and target texts
        src_texts = [item['translation'][self.src_lang] for item in batch]
        tgt_texts = [item['translation'][self.tgt_lang] for item in batch]
        
        # Tokenize and encode
        src_encoded = self._tokenize_and_encode(src_texts)
        tgt_encoded = self._tokenize_and_encode(tgt_texts)

        # Pad the source and target sequences
        src_encoded['input_ids'] = pad_sequence(src_encoded['input_ids'], batch_first=True)
        tgt_encoded['input_ids'] = pad_sequence(tgt_encoded['input_ids'], batch_first=True)
        
        return {
            'src': {
                'input_ids': src_encoded['input_ids'],
                'attention_mask': src_encoded['attention_mask']
            },
            'tgt': {
                'input_ids': tgt_encoded['input_ids'],
                'attention_mask': tgt_encoded['attention_mask']
            }
        } 