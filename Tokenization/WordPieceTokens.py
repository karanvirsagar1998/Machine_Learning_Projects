# WordPiece (BERT tokenizer), Used by BERT, DistilBERT, ALBERT
# We don’t implement WordPiece — we use a pretrained tokenizer (BERT tokenizer) from Hugging Face.
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

text = "I am playing football."

# tokens (WordPiece output)
tokens = tokenizer.tokenize(text)
print(tokens)

# ids (numbers model uses)
ids = tokenizer.encode(text, add_special_tokens=True)
print(ids)

# full output used for models
out = tokenizer(
    text,
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    max_length=16,
    return_tensors="pt"   # use "tf" for TensorFlow
)

print(out["input_ids"])
print(out["attention_mask"])