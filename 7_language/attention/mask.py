"""
BERT Masked Language Modeling and Attention Visualization

This script demonstrates BERT's masked language modeling capabilities and attention mechanisms.
It uses a pre-trained BERT model to predict masked tokens in text and can visualize attention
patterns across different layers and heads of the transformer.

Features:
- Masked language modeling: Predicts the most likely tokens for [MASK] positions
- Multiple mask support: Handles sentences with multiple [MASK] tokens
- Attention visualization: Generates diagrams showing attention weights between tokens
- Multi-layer analysis: Processes all 12 BERT layers and 12 attention heads per layer

Usage:
    python mask.py
    # Enter text with [MASK] tokens when prompted
    # Examples: 
    #   "The [MASK] is sleeping"
    #   "I [MASK] to [MASK] every day"
    #   "The [MASK] is [MASK] in the garden"

Requirements:

    - torch: PyTorch for model inference
    - transformers: Hugging Face transformers library
    - pillow: PIL for image generation (attention diagrams)

Example Output:
    Input: "I [MASK] to [MASK] every day"
    
    Found 2 mask token(s). Generating predictions:
    
    Mask 1 (position 1):
        1. I want to [MASK] every day
        2. I have to [MASK] every day
        3. I try to [MASK] every day
    
    Mask 2 (position 3):
        1. I [MASK] to run every day
        2. I [MASK] to work every day
        3. I [MASK] to him every day

Note: Set VISUALIZE_ATTENTIONS = True to enable attention diagram generation
(creates PNG files for each layer/head combination).
"""

import sys
import torch

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, BertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Whether to generate attention visualization diagrams
VISUALIZE_ATTENTIONS = False

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200

VISUALIZE_ATTENTIONS = False


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="pt")
    mask_indices = get_mask_token_indices(tokenizer.mask_token_id, inputs)
    if not mask_indices:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = BertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions for each mask position
    print(f"\nFound {len(mask_indices)} mask token(s). Generating predictions:\n")
    
    for mask_idx, mask_position in enumerate(mask_indices):
        print(f"Mask {mask_idx + 1} (position {mask_position}):")
        mask_token_logits = result.logits[0, mask_position]
        top_tokens = torch.topk(mask_token_logits, K).indices
        
        for i, token in enumerate(top_tokens):
            # Create a copy of the input_ids and replace only the current mask token
            input_ids_copy = inputs["input_ids"][0].clone()
            input_ids_copy[mask_position] = token
            # Keep other mask tokens as [MASK] in the output
            for other_mask_pos in mask_indices:
                if other_mask_pos != mask_position:
                    input_ids_copy[other_mask_pos] = tokenizer.mask_token_id
            prediction_text = tokenizer.decode(input_ids_copy, skip_special_tokens=False)
            # Clean up special tokens but keep [MASK]
            prediction_text = prediction_text.replace(tokenizer.cls_token, "").replace(tokenizer.sep_token, "").strip()
            
            # Restore original casing by mapping back to the original text
            original_words = text.split()
            prediction_words = prediction_text.split()
            
            # Create a mapping to restore casing
            result_words = []
            for j, word in enumerate(prediction_words):
                if j < len(original_words):
                    # If the original word was capitalized, keep the first letter capitalized
                    if original_words[j][0].isupper() and word[0].islower():
                        word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
                    result_words.append(word)
                else:
                    result_words.append(word)
            
            prediction_text = " ".join(result_words)
            print(f"     {i+1}. {prediction_text}")
        print()
    

    # Visualize attentions
    if VISUALIZE_ATTENTIONS:
        visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_indices(mask_token_id, inputs):
    """
    Return a list of indices where the `mask_token_id` appears in the `inputs`.
    Returns an empty list if no mask tokens are found.
    """

    input_ids = inputs["input_ids"][0]
    mask_indices = []
    for idx, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            mask_indices.append(idx)
    return mask_indices


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """

    gray = int(255 * attention_score)
    return (gray, gray, gray)


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attention_weights = attentions[layer_idx][0][head_idx].detach().numpy()
            generate_diagram(
                layer_idx + 1,
                head_idx + 1,
                tokens,
                attention_weights,
            )
    return


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """

    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
