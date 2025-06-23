---

# PixelRNN - Autoregressive Image Generation using PyTorch

This project is a PyTorch-based implementation of **PixelRNN**, an autoregressive generative model that predicts pixels in an image sequentially. The model is trained to generate images pixel-by-pixel using **RowLSTM** layers, with each pixel treated as a classification problem over 256 intensity levels.

---

## 🧠 Objective

To implement and train a PixelRNN-based architecture that learns the distribution of natural images by modeling pixel dependencies row-wise, enabling the generation of coherent synthetic images.

---

## 🗂 Dataset

* **Source**: Custom RGB image dataset loaded using `torchvision.datasets.ImageFolder`
* **Preprocessing**:

  * Resized to `32x32`
  * Converted to grayscale (if needed)
  * Pixel values quantized to the range `[0, 255]`
  * Transformed using:

    ```python
    transforms.Compose([
        transforms.ToTensor(),
        Lambda(lambda x: (x * 255).long())
    ])
    ```

---

## 🏗 Model Architecture

* **Model**: PixelRNN with RowLSTM layers
* **Input Channels**: 3 (RGB)
* **Hidden Dimension**: 64
* **Output Classes**: 256 (each pixel intensity level)
* **Architecture Details**:

  * Initial 7x7 convolution
  * 4 stacked **RowLSTM** layers (autoregressive modeling across rows)
  * Final 1x1 convolution mapping to output classes (per color channel)

---

## ⚙️ Training Configuration

* **Loss Function**: `CrossEntropyLoss` (applied per channel)
* **Optimizer**: Adam
* **Learning Rate**: 0.001
* **Batch Size**: 16
* **Epochs**: 5
* **Device**: CUDA (GPU)

---

## 📉 Training Results

| Epoch | Average Loss |
| ----- | ------------ |
| 1     | \~11.718     |
| 2     | \~9.312      |
| 3     | \~7.579      |

> The training loss steadily decreased, showing the model's learning progress. Generated images qualitatively showed local coherence and color structure, though global image quality can be improved.

---

## 📊 Comparison with Original PixelRNN Paper

| Feature               | Original Paper            | Our Implementation       |
| --------------------- | ------------------------- | ------------------------ |
| Dataset               | CIFAR-10, MNIST, ImageNet | Custom RGB dataset       |
| Architecture          | Diagonal BiLSTM / RowLSTM | RowLSTM only             |
| Output Representation | Discrete Softmax          | Discrete Softmax         |
| Evaluation Metric     | NLL (bits/dim)            | Cross-Entropy Loss       |
| Performance           | \~3.0 bits/dim (CIFAR-10) | Qualitative results only |

Due to computational constraints, bits-per-dimension evaluation was not included.

---

## 🔍 Key Takeaways

* **RowLSTM** is effective in capturing pixel dependencies in an autoregressive manner.
* Pixel-level classification using softmax is powerful but **computationally heavy** at inference.
* The project replicates core ideas from the original PixelRNN paper.
* **Areas for Improvement**:

  * Incorporating Diagonal BiLSTM for full receptive field
  * Training on full CIFAR-10 or ImageNet for better generalization
  * Adding residual connections or deeper networks

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --epochs 5 --batch_size 16

# Generate images
python generate.py --model_path saved_model.pth
```

---

## 📚 References

* [Van den Oord et al., Pixel Recurrent Neural Networks, 2016](https://arxiv.org/abs/1601.06759)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---
