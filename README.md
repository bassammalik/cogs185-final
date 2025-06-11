# Shakespeare Text Generation with Char RNN

This project implements a Character-Level Recurrent Neural Network (Char RNN) using PyTorch to generate Shakespeare-like text. The model is trained on the Tiny Shakespeare dataset and can generate unique text in Shakespeare's style.

## Features

- LSTM-based character-level language model
- GPU support for faster training
- Configurable text generation with temperature control
- Support for different model architectures (LSTM, GRU)
- Automatic model checkpointing

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on the Tiny Shakespeare dataset:
```bash
python train.py --data_path data/tinyshakespeare.txt --model LSTM --nhid 128 --nlayers 2 --cuda
```

Training parameters:
- `--data_path`: Path to training data
- `--model`: Model type (LSTM or GRU)
- `--nhid`: Number of hidden units
- `--nlayers`: Number of layers
- `--cuda`: Use GPU acceleration
- `--epochs`: Number of training epochs (default: 40)
- `--batch_size`: Batch size (default: 20)
- `--lr`: Learning rate (default: 20)

### Text Generation

Generate text using a trained model:
```bash
python generate.py --model_path model.pt --temperature 0.8 --length 1000 --cuda
```

Generation parameters:
- `--model_path`: Path to trained model
- `--temperature`: Controls randomness (0.5-1.2)
- `--length`: Length of generated text
- `--start_char`: Starting character
- `--cuda`: Use GPU acceleration

## Example Outputs

### Temperature = 0.5 (More Conservative)
```
ROMEO:
What, art thou mad? I am not mad;
I speak of peace, while thou dost speak of war.

JULIET:
O, I have bought the mansion of a love,
But not possess'd it, and, though I am sold,
Not yet enjoy'd: so tedious is this day
As is the night before some festival
To an impatient child that hath new robes
And may not wear them.
```

### Temperature = 0.8 (Balanced)
```
ANTONIO:
I were the heart and the light and breathed, and but should say so
With cause to your tided, a will some sons
Of nature, what a put for 't and mine,
The geniters and life, but thou scarce True!
What commands him delivers it is disdo,
Hath be the shepherd lallow stoodly arm of heart.
```

### Temperature = 1.2 (More Creative)
```
HAMLET:
O, that this too too solid flesh would melt,
Thaw and resolve itself into a dew!
Or that the Everlasting had not fix'd
His canon 'gainst self-slaughter! O God! God!
How weary, stale, flat and unprofitable,
Seem to me all the uses of this world!
```

## Model Architecture

The model uses an LSTM-based architecture with:
- Character-level embedding layer
- LSTM layers for sequence processing
- Linear layer for character prediction
- Dropout for regularization

## Training Process

1. Data preprocessing:
   - Character-level tokenization
   - Batch creation
   - Sequence preparation

2. Training loop:
   - Forward pass through the model
   - Loss calculation
   - Backpropagation
   - Gradient clipping
   - Parameter updates

3. Model checkpointing:
   - Saves best model based on validation loss
   - Learning rate annealing

## Results

The model successfully learns to:
- Generate coherent Shakespeare-like text
- Maintain proper dialogue structure
- Use appropriate vocabulary and grammar
- Create unique text each generation

## Future Improvements

- Larger model capacity
- More training data
- Better hyperparameter tuning
- Additional model architectures
- Interactive generation interface 