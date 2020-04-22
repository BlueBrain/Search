import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


class SBioBERT(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.biobert_model = AutoModelWithLMHead.from_pretrained("gsarti/biobert-nli").bert
        self.tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")

    def preprocess_sentence(self, sentence):
        # Add the special tokens.
        marked_text = "[CLS] " + sentence + " [SEP]"

        # Split the sentence into tokens.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Map the token strings to their vocabulary indices.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors

    def encode(self, sentences):
        preprocessed_sentences = [self.preprocess_sentence(sentence)
                                  for sentence in sentences]

        results = []
        for tokens_tensor, segments_tensors in preprocessed_sentences:
            with torch.no_grad():
                encoded_layers, _ = self.biobert_model(tokens_tensor, segments_tensors)
                sentence_encoding = encoded_layers[-1].squeeze().mean(axis=0)
                results.append(sentence_encoding.detach().cpu().numpy())

        return results
