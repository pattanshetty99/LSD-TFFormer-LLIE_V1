# LSD-TFFormer-LLIE_V1

## Low-Light Image Enhancement using Hierarchical Swin Transformer with LayerScale and DropPath

---

## 📌 Overview

**LSD-TFFormer-LLIE** is a hierarchical Transformer-based deep learning model designed for **Low-Light Image Enhancement (LLIE)**.

This implementation includes:

- ✅ 512×512 full-resolution training
- ✅ Hierarchical Encoder–Decoder architecture
- ✅ Swin-style Window Attention with Shifted Windows
- ✅ LayerScale stabilization
- ✅ DropPath (Stochastic Depth)
- ✅ Multi-loss training (L1 + SSIM + Perceptual)
- ✅ Automatic Mixed Precision (AMP)
- ✅ Cosine Learning Rate Scheduler
- ✅ PSNR & SSIM evaluation pipeline
- ✅ Test image inference & saving

The architecture is designed for high-resolution enhancement tasks and is optimized for modern GPUs.

---

## 🧠 Model Architecture

### Resolution Flow

| Stage | Resolution |
|-------|------------|
| Input | 512 × 512 |
| Encoder Down1 | 256 × 256 |
| Encoder Down2 | 128 × 128 |
| Transformer Bottleneck | 128 × 128 |
| Decoder Up1 | 256 × 256 |
| Output | 512 × 512 |

### Key Features

- **Swin-style Window Attention**
- **Shifted Window Mechanism**
- **LayerNorm (Pre-Norm)**
- **LayerScale for stable deep training**
- **DropPath regularization**
- **Skip Connections (U-Net style)**

---

## 📂 Project Structure

```
LSD-TFFormer-LLIE/
│
├── models/
│   ├── lsd_tf_former.py
│   ├── swin_blocks.py
│   ├── layers.py
│
├── datasets/
│   ├── llie_dataset.py
│
├── losses/
│   ├── perceptual.py
│   ├── ssim.py
│
├── utils/
│   ├── metrics.py
│   ├── checkpoint.py
│   ├── scheduler.py
│
├── train.py
├── validate.py
├── test.py
├── evaluation.py
├── config.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/LSD-TFFormer-LLIE_V1.git
cd LSD-TFFormer-LLIE
```

### 2️⃣ Install PyTorch (CUDA Recommended)

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Structure

Organize dataset as:

```
data/
│
├── train/
│   ├── low/
│   └── high/
│
├── val/
│   ├── low/
│   └── high/
│
├── test/
│   ├── low/
│   └── high/
```

Each low-light image must have a corresponding ground-truth image with the same filename.

---

## 🚀 Training

Edit `config.py` if needed.

Start training:

```bash
python train.py
```

During training:
- Best model saved in `checkpoints/best.pth`
- PSNR displayed per epoch

---

## 🧪 Validation

Validation runs automatically during training.

Standalone validation:

```bash
python validate.py
```

---

## 🔍 Testing (Inference Only)

To enhance test images:

```bash
python test.py
```

Enhanced images will be saved in:

```
results/
```

---

## 📈 Evaluation (PSNR + SSIM)

To compute quantitative metrics:

```bash
python evaluation.py
```

Output:

```
Average PSNR: XX.XXX dB
Average SSIM: X.XXXX
```

---

## 🏗 Training Configuration

Default configuration (config.py):

- Resolution: 512×512
- Batch Size: 2
- Epochs: 250
- Optimizer: AdamW
- Learning Rate: 2e-4
- Scheduler: CosineAnnealingLR
- Loss:
  - L1
  - SSIM
  - LPIPS (Perceptual)

---

## 📦 Requirements

See `requirements.txt`

Main dependencies:

- torch
- torchvision
- opencv-python
- numpy
- scikit-image
- lpips
- tqdm

---

## 💡 Advanced Features Included

- Swin-style shifted window attention
- LayerScale for stable deep networks
- DropPath for better generalization
- Hierarchical Transformer design
- Memory-efficient bottleneck at 128×128

---

## 🔮 Future Improvements

- Multi-stage transformer refinement
- Frequency-domain loss
- GAN-based enhancement
- NTIRE competition tuning
- Large-scale dataset training

---

## 📜 License

This project is released for research and academic purposes.

---

## 🙌 Acknowledgments

Inspired by hierarchical Transformer-based image restoration frameworks and Swin Transformer architecture principles.

---

## 👩‍💻 Author

Your Name  
GitHub: https://github.com/yourusername

---

If you use this repository for research, please consider citing appropriately.

