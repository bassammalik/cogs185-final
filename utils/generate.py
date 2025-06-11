import argparse
import torch
import torch.nn as nn
import numpy as np
from model import RNNModel

# Generation settings
parser = argparse.ArgumentParser(description='PyTorch Char RNN Generation')
parser.add_argument('--model_path', type=str, required=True,
                    help='path to model checkpoint')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--length', type=int, default=1000,
                    help='length of generated text')
parser.add_argument('--start_char', type=str, default='\n',
                    help='starting character')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load the model
print('Loading model from', args.model_path)
with open(args.model_path, 'rb') as f:
    model = torch.load(f, map_location=torch.device('cpu'))
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

def generate():
    hidden = model.init_hidden(1)
    if args.cuda:
        hidden = (hidden[0].cuda(), hidden[1].cuda()) if isinstance(hidden, tuple) else hidden.cuda()
    
    # Convert start char to tensor
    start_char = args.start_char
    if len(start_char) > 1:
        start_char = start_char[0]  # Take first character if multiple provided
    
    # Get the character mapping from the model's encoder
    # Use the exact same character set as in training
    chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    
    # Print vocabulary size for debugging
    print(f"Vocabulary size: {len(chars)}")
    print(f"Characters: {''.join(chars)}")
    
    # Ensure the start character is in our vocabulary
    if start_char not in char_to_ix:
        print(f"Warning: Start character '{start_char}' not in vocabulary. Using newline instead.")
        start_char = '\n'
    
    input = torch.tensor([[char_to_ix[start_char]]], dtype=torch.long)
    if args.cuda:
        input = input.cuda()
    
    generated_text = start_char
    
    with torch.no_grad():
        for i in range(args.length):
            output, hidden = model(input, hidden)
            
            # Sample from the network as a multinomial distribution
            output_dist = output[0, -1].div(args.temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Add predicted character to string and use as next input
            predicted_char = chars[top_i]
            generated_text += predicted_char
            input = torch.tensor([[top_i]], dtype=torch.long)
            if args.cuda:
                input = input.cuda()
    
    return generated_text

# Generate text
print("\nGenerating", args.length, "characters of text...\n")
print("Generated Text:")
print("-" * 50)
print(generate())
print("-" * 50) 