import subprocess
import time
import json
from datetime import datetime

def run_command(command):
    print(f"\nRunning: {command}")
    start_time = time.time()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()
    return {
        'command': command,
        'stdout': stdout.decode(),
        'stderr': stderr.decode(),
        'time_taken': end_time - start_time
    }

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

# Create results directory
subprocess.run('mkdir -p results', shell=True)

# 1. Generate text with different temperatures
print("\n=== Generating text with different temperatures ===")
generation_results = {}

# Conservative generation
result = run_command('python generate.py --model_path model.pt --temperature 0.5 --length 500 --start_char "ROMEO"')
generation_results['temp_0.5'] = result

# Balanced generation
result = run_command('python generate.py --model_path model.pt --temperature 0.8 --length 500 --start_char "HAMLET"')
generation_results['temp_0.8'] = result

# Creative generation
result = run_command('python generate.py --model_path model.pt --temperature 1.2 --length 500 --start_char "MACBETH"')
generation_results['temp_1.2'] = result

save_results(generation_results, 'results/generation_results.json')

# 2. Train models with different architectures (using fewer epochs for CPU)
print("\n=== Training models with different architectures ===")
training_results = {}

# LSTM with 1 layer
result = run_command('python train.py --data_path data/tinyshakespeare.txt --model LSTM --nhid 128 --nlayers 1 --epochs 10 --save results/model_lstm_1layer.pt')
training_results['lstm_1layer'] = result

# LSTM with 2 layers
result = run_command('python train.py --data_path data/tinyshakespeare.txt --model LSTM --nhid 128 --nlayers 2 --epochs 10 --save results/model_lstm_2layer.pt')
training_results['lstm_2layer'] = result

# GRU with 2 layers
result = run_command('python train.py --data_path data/tinyshakespeare.txt --model GRU --nhid 128 --nlayers 2 --epochs 10 --save results/model_gru_2layer.pt')
training_results['gru_2layer'] = result

save_results(training_results, 'results/training_results.json')

# 3. Test different hyperparameters
print("\n=== Testing different hyperparameters ===")
hyperparam_results = {}

# Different learning rates
result = run_command('python train.py --data_path data/tinyshakespeare.txt --model LSTM --nhid 128 --nlayers 2 --lr 10 --epochs 10 --save results/model_lr10.pt')
hyperparam_results['lr_10'] = result

result = run_command('python train.py --data_path data/tinyshakespeare.txt --model LSTM --nhid 128 --nlayers 2 --lr 30 --epochs 10 --save results/model_lr30.pt')
hyperparam_results['lr_30'] = result

# Different batch sizes
result = run_command('python train.py --data_path data/tinyshakespeare.txt --model LSTM --nhid 128 --nlayers 2 --batch_size 10 --epochs 10 --save results/model_bs10.pt')
hyperparam_results['bs_10'] = result

result = run_command('python train.py --data_path data/tinyshakespeare.txt --model LSTM --nhid 128 --nlayers 2 --batch_size 40 --epochs 10 --save results/model_bs40.pt')
hyperparam_results['bs_40'] = result

save_results(hyperparam_results, 'results/hyperparam_results.json')

print("\nAll experiments completed! Results saved in the 'results' directory.") 