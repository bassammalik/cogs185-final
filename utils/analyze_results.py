import json
import matplotlib.pyplot as plt
import numpy as np
import re
import os

def extract_loss_perplexity(text):
    # Extract loss and perplexity from training output
    loss_pattern = r'valid loss (\d+\.\d+)'
    ppl_pattern = r'valid ppl (\d+\.\d+)'
    
    losses = [float(x) for x in re.findall(loss_pattern, text)]
    perplexities = [float(x) for x in re.findall(ppl_pattern, text)]
    
    return losses, perplexities

def plot_training_curves(results_file, title):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    plt.figure(figsize=(12, 6))
    
    for model_name, result in results.items():
        if 'stdout' in result:
            losses, perplexities = extract_loss_perplexity(result['stdout'])
            if losses:
                epochs = range(1, len(losses) + 1)
                plt.plot(epochs, losses, label=f'{model_name} (Loss)')
                plt.plot(epochs, perplexities, label=f'{model_name} (Perplexity)', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png')
    plt.close()

def extract_text_blocks(file_content):
    # Split the file into blocks for each temperature
    blocks = file_content.split('Loading model from model_gru_2layer.pt')
    texts = []
    for block in blocks:
        if 'Generated Text:' in block:
            text = block.split('Generated Text:\n--------------------------------------------------\n')[1].split('\n--------------------------------------------------')[0]
            texts.append(text)
    return texts

def analyze_lstm_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    analysis = {}
    for temp, result in results.items():
        if 'stdout' in result:
            text = result['stdout'].split('Generated Text:\n--------------------------------------------------\n')[1].split('\n--------------------------------------------------')[0]
            analysis[temp] = {
                'length': len(text),
                'unique_chars': len(set(text)),
                'avg_word_length': np.mean([len(word) for word in text.split()]),
                'time_taken': result['time_taken'],
                'text': text
            }
    return analysis

def analyze_gru_results(results_file):
    with open(results_file, 'r') as f:
        file_content = f.read()
    texts = extract_text_blocks(file_content)
    analysis = {}
    for i, temp in enumerate(['temp_0.5', 'temp_0.8', 'temp_1.2']):
        if i < len(texts):
            text = texts[i]
            analysis[temp] = {
                'length': len(text),
                'unique_chars': len(set(text)),
                'avg_word_length': np.mean([len(word) for word in text.split()]),
                'text': text
            }
    return analysis

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Analyze LSTM results
lstm_analysis = analyze_lstm_results('results/generation_results.json')
# Analyze GRU results
gru_analysis = analyze_gru_results('results/gru_generation_results.json')

# Print comparative analysis
print("\nComparative Analysis: LSTM vs GRU\n" + "="*40)
for temp in ['temp_0.5', 'temp_0.8', 'temp_1.2']:
    print(f"\nTemperature {temp.split('_')[1]}:")
    print("-"*30)
    print("LSTM:")
    if temp in lstm_analysis:
        la = lstm_analysis[temp]
        print(f"  Text length: {la['length']} characters")
        print(f"  Unique characters: {la['unique_chars']}")
        print(f"  Average word length: {la['avg_word_length']:.2f}")
        print(f"  Sample: {la['text'][:120]}...")
    else:
        print("  No data.")
    print("GRU:")
    if temp in gru_analysis:
        ga = gru_analysis[temp]
        print(f"  Text length: {ga['length']} characters")
        print(f"  Unique characters: {ga['unique_chars']}")
        print(f"  Average word length: {ga['avg_word_length']:.2f}")
        print(f"  Sample: {ga['text'][:120]}...")
    else:
        print("  No data.")

# Save combined analysis
with open('results/combined_analysis.json', 'w') as f:
    json.dump({'lstm': lstm_analysis, 'gru': gru_analysis}, f, indent=2)

print("\nFull comparative analysis saved to results/combined_analysis.json")

# Analyze generation results
print("Analyzing generation results...")
gen_analysis = analyze_generation_results('results/generation_results.json')

# Print analysis
print("\nGeneration Analysis:")
print("-------------------")
for temp, stats in gen_analysis.items():
    print(f"\nTemperature {temp}:")
    print(f"Text length: {stats['length']} characters")
    print(f"Unique characters: {stats['unique_chars']}")
    print(f"Average word length: {stats['avg_word_length']:.2f}")
    print(f"Generation time: {stats['time_taken']:.2f} seconds")
    print("\nSample text:")
    print(stats['text'][:200] + "...")

# Save analysis
with open('results/analysis.json', 'w') as f:
    json.dump({
        'generation_analysis': gen_analysis
    }, f, indent=2)

print("\nAnalysis complete! Results saved in 'results/analysis.json'") 