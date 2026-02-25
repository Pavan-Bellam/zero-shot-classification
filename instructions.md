### **Zero-Shot Text Classification Assignment**

**Objective:**  
Build and train a zero-shot text classification model.

### **1. Project Structure and Setup**

- **Directory Structure:**
  Organize your code and data as follows:
  ```
  ├── data/
  │   └── synthetic_data.json
  ├── scripts/
  │   ├── train.py
  ├── model.py  # Original BiEncoderModel code
  ├── dataset.py  # Dataset implementation
  ├── config.yaml
  └── README.md
  ```
- **Configuration Files:**
  Provide a `config.yaml` for easy model and training configuration:
  ```yaml
  model:
    name: "bert-base-uncased"
    max_num_labels: 5

  training:
    num_steps: 1000
    batch_size: 8
    learning_rate: 2e-5
    optimizer: "adamw"

  data:
    synthetic_data_path: "data/synthetic_data.json"
  ```

### **2. Model Development**

- **BiEncoder Model Architecture:**
  Implement the following model to handle both text and label encoding:
  
  ```python
  import torch
  import torch.nn as nn
  from transformers import AutoModel, AutoTokenizer

  class BiEncoderModel(nn.Module):
      def __init__(self, model_name, max_num_labels):
          super(BiEncoderModel, self).__init__()
          # Shared encoder for both text and labels
          self.shared_encoder = AutoModel.from_pretrained(model_name)
          self.tokenizer = AutoTokenizer.from_pretrained(model_name)
          self.max_num_labels = max_num_labels  # Maximum labels per sample

      def encode(self, texts_or_labels):
          """
          Encodes a list of texts or labels using the shared encoder.
          """
          inputs = self.tokenizer(texts_or_labels, return_tensors='pt', padding=True, truncation=True)
          outputs = self.shared_encoder(**inputs)
          # mask aware pooling
          # last_hidden_state: [B, seq_len, D]
          att_mask = inputs['attention_mask'].unsqueeze(-1)
          return (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(1)

      def forward(self, texts, batch_labels):
          """
          texts: List of input texts with batch size B
          batch_labels: List of lists containing labels for each text
          """
          B = len(texts)

          # Flatten all labels in the batch
          all_labels = [label for labels in batch_labels for label in labels]
          label_embeddings = self.encode(all_labels)  # Shape: [Num_unique_labels, D]

          # Encode texts
          text_embeddings = self.encode(texts)  # Shape: [B, D]
          # Prepare to recover batch structure
          label_counts = [len(labels) for labels in batch_labels]
          max_num_label = self.max_num_labels
          padded_label_embeddings = torch.zeros(B, max_num_label, label_embeddings.size(-1))
          mask = torch.zeros(B, max_num_label, dtype=torch.bool)

          current = 0
          for i, count in enumerate(label_counts):
              if count > 0:
                  end = current + count
                  padded_label_embeddings[i, :count, :] = label_embeddings[current:end]
                  mask[i, :count] = 1
                  current = end

          # Compute similarity scores between text and each label
          # text_embeddings: [B, D]
          # padded_label_embeddings: [B, max_num_label, D]
          # scores: [B, max_num_label]
          scores = torch.bmm(padded_label_embeddings, text_embeddings.unsqueeze(2)).squeeze(2)
          scores = torch.sigmoid(scores)

          return scores, mask

      @torch.no_grad()
      def forward_predict(self, texts, labels):
          """
          texts: List of input texts
          labels: List of labels corresponding to the texts
          Returns:
              List of JSON objects with label scores for each text
          """
          scores, mask = self.forward(texts, labels)
          results = []
          for i, text in enumerate(texts):
              text_result = {}
              for j, label in enumerate(labels[i]):
                  if mask[i, j]:
                      text_result[label] = float(f"{scores[i, j].item():.2f}")
              results.append({"text": text, "scores": text_result})
          return results

  # Example Usage
  if __name__ == "__main__":
      model_name = "bert-base-uncased"
      max_num_labels = 5
      model = BiEncoderModel(model_name, max_num_labels)

      texts = ["I love machine learning.", "Deep learning models are powerful."]
      batch_labels = [
          ["AI", "Machine Learning"],
          ["Deep Learning", "Neural Networks", "AI"]
      ]

      # Forward pass
      scores, mask = model(texts, batch_labels)
      print("Scores:", scores)
      print("Mask:", mask)

      # Prediction with JSON output
      predictions = model.forward_predict(texts, batch_labels)
      print("Predictions:", predictions)
  ```

  **To-Do Items:**
  - Implement the loss function (e.g., sigmoid cross-entropy).
  - Modify model architecture as necessary.
  - Integrate with Hugging Face Hub for ease of saving and sharing.
  
  ```python
  # Save pretrained components
  model.save_pretrained("model_dir") # save weights, tokenizer, and configs

  # Push to Hugging Face Hub
  model.push_to_hub("your-username/bi-encoder-model")

  # Load the model
  from model import BiEncoderModel

  model = BiEncoderModel.from_pretrained("your-username/bi-encoder-model")
  ```

### **3. Data Processing**

- **Dataset Class Implementation:**
  Develop a dataset class to handle data loading, preprocessing, and negative sampling.

- **Negative Sampling:**
  Implement negative sampling to increase the robustness of the model.
  
  **Example: Negative Sampling Implementation**
  ```python
  import random

  def negative_sampling(batch_labels, all_labels, max_num_negatives=10):
      num_negatives = random.randint(1, max_num_negatives)
      negative_samples = []
      for labels in batch_labels:
          neg = random.sample([l for l in all_labels if l not in labels], num_negatives)
          negative_samples.append(neg)
      return negative_samples
  ```

### **4. Training Setup**

- **GPU Training:**
  To speed up the training process, ensure that your model runs on a GPU. You can achieve this by moving your model and data to the GPU using PyTorch's `.to('cuda')` method if a GPU is available.
- 
- **Training Script (`scripts/train.py`):**
  Implement a training script that loads the configuration file, processes the synthetic data, and trains the BiEncoder model.

### **5. Data Generation**

- **Generate Synthetic Training Data:**
  Use an LLM (eg. gpt4o-mini) to generate diverse synthetic examples for training.
  
  **Example Data Format:**
  ```json
  [
    {"text": "The stock market crashed yesterday.", "labels": ["Finance", "Economy"]},
    {"text": "A new species of bird was discovered in the Amazon.", "labels": ["Biology", "Environment", "Animals"]}
  ]
  ```
  
  **Pseudo Code for Data Generation:**
  ```python
  prompts = "Generate a sentence and assign 2-3 diverse labels:"
  synthetic_data = llm.generate(prompts, num_samples=1000)
  # Parse LLM output into {"text": ..., "labels": [...]} format
  ```

### **6. Deliverables**

- **Full Implementation:**
  - Complete implementation of the model, data processing, training script, and setup as described in the assignment.

- **Trained Model Uploaded to Hugging Face:**
  - After training the model, upload it to the Hugging Face Hub. The model should be uploaded to a private repository and accessible only to authorized users.
  - You may train the model on Google Colab and provide Colab instructions in the README for ease of use.

- **Bonus Tasks:** (*Optional*)
  - **Novel/Original Architecture:** Implement a novel or original model architecture to improve performance.
  - **Diverse Variants:** Implement diverse model variants, such as:
    - **Late Interaction:** [Late Interaction Paper](https://arxiv.org/abs/2004.12832)
    - **Polyencoder:** [Polyencoder Paper](https://arxiv.org/abs/1905.01969)
  - **Benchmark Setup:** Establish benchmarks for model performance and training speed, and compare different model variants.
  - **Pros and Cons Analysis:** Analyze the pros and cons of different approaches used in zero-shot text classification.
  - **Literature Review:** Write a concise literature review of zero-shot text classification approaches, including LLM-based methods, prompting strategies, and BiEncoder architectures.
  - **Synthetic Data Generation Script:** Implement a script for generating text classification synthetic data using the OpenAI API.
  - **LLM as a Judge for Evaluation:** A script that uses an LLM (eg gpt-4o) to evaluate the model's predictions.