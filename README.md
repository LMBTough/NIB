# Narrowing Information Bottleneck (NIB)

This repository contains the official implementation of the **Narrowing Information Bottleneck (NIB)** algorithm proposed in the paper:  
**Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability**, under review for ICLR 2025.

## Abstract

The task of interpreting multimodal image-text representations is gaining increasing attention with models like CLIP excelling at learning associations between images and text. Despite advancements, ensuring the interpretability of these models is crucial for their safe application in real-world scenarios such as healthcare. We propose the **Narrowing Information Bottleneck Theory (NIB)**, which re-engineers the bottleneck method from the ground up to satisfy modern attribution axioms, offering a robust solution for improving the interpretability of multimodal models.

## Key Features

- **Improved Interpretability**: The NIB method improves interpretability in both image and text representations.
- **No Randomness**: Unlike existing bottleneck methods, NIB eliminates randomness and hyperparameter sensitivity, offering more deterministic outcomes.
- **Attribution Accuracy**: Our approach improves image interpretability by 9% and text interpretability by 58.83%.
- **Increased Efficiency**: NIB increases processing speed by 63.95% compared to other methods.

## Installation

## Usage
### Pretrained Models
We utilize the CLIP model with a Vision Transformer (ViT-B/32) as the visual encoder for multimodal image-text representation tasks. Make sure to have the pretrained CLIP model ready before running the experiments.

### Running the Code
To evaluate NIB on datasets such as Conceptual Captions, ImageNet, and Flickr8k, use the following command:

```bash
python main.py --dataset ConceptualCaptions --model ViT-B-32
```

Replace `ConceptualCaptions` with the desired dataset name, and `ViT-B-32` with the model of your choice.


###  Hyperparameters
The default settings for the number of iterations (num_steps) and layer number are provided below. You can modify them based on your computational needs.

num_steps: 10
layer_number: 9
To modify these parameters, pass them as arguments:

```bash
python main.py --num_steps 15 --layer_number 9
```

###  Datasets
The experiments in the paper are conducted on the following datasets:

Conceptual Captions: A large-scale image-text alignment dataset.
ImageNet: A widely used image classification dataset.
Flickr8k: A smaller image-text alignment dataset.
You can download these datasets from their respective sources and provide their paths when running the experiments.

###  Evaluation
We use the Confidence Drop and Confidence Increase metrics to evaluate the performance of the attribution algorithms. These metrics assess the degradation or improvement in model performance after modifying the key features based on attribution scores.

```bash
python evaluate.py --dataset ConceptualCaptions --metrics ConfidenceDrop
```

