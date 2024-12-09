import torch
import unittest

from model.transformer import TransformerEncoder, TransformerDecoder


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.d_model = 128
        self.n_heads = 8
        self.d_ff = 256
        self.n_layers = 4
        self.dropout = 0.1
        self.norm_order = "post"

    def test_transformer(self):
        encoder = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.n_layers, self.dropout, self.norm_order)
        decoder = TransformerDecoder(self.d_model, self.n_heads, self.d_ff, self.n_layers, self.dropout, self.norm_order)

        src = torch.randn(10, 50, self.d_model)
        tgt = torch.randn(10, 50, self.d_model)

        src_mask = None
        tgt_mask = None
        memory_mask = None

        encoder_output = encoder(src, src_mask)
        decoder_output = decoder(tgt, encoder_output, tgt_mask, memory_mask)

        self.assertEqual(encoder_output.shape, (10, 50, self.d_model))
        self.assertEqual(decoder_output.shape, (10, 50, self.d_model))

if __name__ == "__main__":
    unittest.main()
