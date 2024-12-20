# Neural Dependency Parser

This repository provides an implementation of a neural dependency parser using PyTorch. The parser leverages pretrained word embeddings, a feedforward neural network, and transition-based parsing to construct dependency trees for natural language sentences.

The parser is equipped with components for training, evaluating, and parsing sentences in minibatches. Below is an example of its use in training and evaluating a model:

---

## What This Repository Does

1. **Dependency Parsing**: Implements a transition-based parser with "Shift", "Left-Arc", and "Right-Arc" transitions to construct syntactic dependencies.

2. **Neural Network Model**: Uses a feedforward neural network with a single hidden layer, dropout regularization, and pretrained embeddings.

3. **Minibatch Parsing**: Processes multiple sentences efficiently in batches.

4. **Model Training and Evaluation**: Includes training loops with cross-entropy loss and UAS (Unlabeled Attachment Score) evaluation on development and test datasets.

---

## Installation

### Using Conda
This package is distributed with a Conda environment. To install:

```bash
conda env create -f environment.yml
conda activate dependency-parser
```

## How to Run the Code

## Step 1: Prepare Your Dataset

- Place your datasets (`train.conll`, `dev.conll`, `test.conll`) in the `src/data` directory.
- Ensure pretrained embeddings (e.g., `en-cw.txt`) are also present in the `src/data` directory.

## Step 2: Execute the Training Script

Run the following command to train the parser model:

```bash
python run.py --device <device>
```

#### Example (using GPU):

```bash
python run.py --device gpu
```

#### **Key Parameters**

The following parameters can be customized when running `run.py`:

| **Parameter** | **Description**                                   | **Default** |
|---------------|---------------------------------------------------|-------------|
| `--device`    | Device to use for computation (`cpu`, `gpu`).     | `cpu`       |
| `--debug`     | Run in debug mode with reduced dataset size.      | `False`     |
| `--compile`   | Compile the model for faster execution.           | `False`     |
| `--backend`   | Compilation backend (e.g., `inductor`).           | `inductor`  |

## Step 3: Training Process

- The script trains the parser over multiple epochs, saving the best model weights based on development set performance.
- Training duration depends on system performance and dataset size.

## Step 4: Testing the Model

The trained model is automatically evaluated on the test dataset after training. Results, including the UAS score, are displayed in the terminal.

#### Output

Upon completing the training and evaluation:
1. Model Weights:
  -  Saved in the `run_results_(soln)/` directory.
2. Parsing Results:
  - Final Unlabeled Attachment Score (UAS) for the test dataset is reported.

## Notes
- Dataset Compatibility: The parser expects input in **CoNLL** format. If using custom datasets, ensure they are preprocessed into this format.
- Pretrained Embeddings: The model requires pretrained embeddings as input. Modify `parser_utils.py` if using custom embeddings.
- Versatility: Extend the model or parser logic by modifying `parser_model.py` or `parser_transitions.py`.

## Acknowledgments

This project was inspired by [Natural Language Processing with Deep Learning](https://online.stanford.edu/courses/xcs224n-natural-language-processing-deep-learning) course provided by Stanford University.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contacts

- LinkedIn: [Enrico Zanetti](https://www.linkedin.com/in/enrico-zanetti/)
