import sys
import os

# Ajoute le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.create_transformator import Transformer


vocab_size = 100  # Size of sentences : here 100 words
seq_len = 10      # Length of sequences
batch_size = 2    # For lack of memory matter....we restrain to 2 batchs

model = Transformer(vocab_size, d_model=32, num_heads=2, num_layers=2, dim_feedforward=64, max_len=seq_len)
input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(input_seq)
print(output.shape)  # Devrait être [batch_size, seq_len, vocab_size]
