# Image-Captioning-with-GRUs
This project generates image captions using a deep learning model combining convolutional and recurrent architectures. The model is trained on the **Flickr8k** dataset and utilizes **EfficientNet**, **GRU**, and **tokenization techniques** to describe image content in natural language.

##  Overview

- Load and clean image captions.
- Preprocess and extract image features using EfficientNet/VGG16/ResNet.
- Tokenize captions and prepare training sequences.
- Train a multimodal model that combines image features and text.
- Generate descriptive captions for unseen images.

---

##  Tools & Libraries

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL
- TQDM
- EfficientNetB7, VGG16, ResNet152
- GRU, Embedding, Dense, Dropout, BatchNormalization, LayerNormalization

---

##  Dataset

- **Dataset Name**: Flickr8k
  https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k
- **Description**: A dataset of 8000 images, each paired with 5 human-written captions.
- **Files Used**:
  - `Flickr8k.token.txt`: Captions for each image.
  - `Flicker8k_Dataset/`: Directory containing `.jpg` image files.

---

##  Pipeline

###  1. Load & Clean Captions
- Remove punctuation, convert to lowercase.
- Keep only alphabetical words.
- Wrap each caption with `startseq` and `endseq` tokens.

###  2. Preprocess Images
- Resize based on model type:
  - EfficientNetB7: `600x600`
  - ResNet152: `448x448`
  - VGG16: `224x224`
- Apply model-specific preprocessing functions.

###  3. Extract Image Features
- Use a pre-trained CNN model (EfficientNet/VGG/ResNet).
- Extract and flatten image features.

###  4. Tokenize & Prepare Captions
- Fit a Keras `Tokenizer` on cleaned captions.
- Convert words to sequences.
- Pad sequences to the max caption length.

###  5. Data Generator
- Custom generator yields batches of:
  - Image features
  - Input word sequences
  - One-hot encoded next words

### 6. Build Model Architecture

**Model Inputs**:
- Image Feature Vector
- Tokenized Caption Sequence

**Model Components**:
- Dense layers with Dropout and BatchNormalization
- GRU layers for caption generation
- Final Dense layer with `softmax` over vocabulary

