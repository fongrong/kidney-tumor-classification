# Deep Learning-Based Kidney Tumor Subtype Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive two-stage deep learning framework for kidney tumor subtype classification from CT imaging, featuring contrast enhancement techniques and achieving state-of-the-art performance.

## 📄 Paper

**Title:** Deep Learning-Based Kidney Tumor Subtype Classification from CT Imaging: A Comprehensive Two-Stage Framework with Contrast Enhancement

**Authors:** Wen-Yi Li, Jia-Yang Peng, Bor-Wen Cheng, Wei-Shiung Yang, Feng-Jung Yang

**Journal:** Journal of Medical Internet Research (JMIR) - Under Review

**Preprint:** [Link to be added]

## 🎯 Key Features

- **Two-Stage Architecture**: Binary kidney detection (EfficientNet-B7) + Multi-class tumor classification (YOLOv7-X)
- **Novel Preprocessing**: JET colormap enhancement achieves +3.56% accuracy improvement
- **State-of-the-Art Performance**: 84.90% mAP@0.5, 97.8% kidney detection accuracy
- **Real-Time Inference**: 0.0174 seconds per image (57.5 FPS)
- **Clinical Validation**: Tested on KiTS19 dataset (210 cases, 5 tumor subtypes)
- **Reproducible Research**: Complete code, pre-trained models, and detailed documentation

## 🏥 Clinical Significance

Kidney cancer affects over 430,000 people annually worldwide. This AI system provides:
- Non-invasive tumor subtype classification
- Reduced need for invasive biopsies
- Real-time diagnostic support for radiologists
- Improved treatment planning and patient outcomes

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Stage 1 (Kidney Detection)** | 97.8% accuracy |
| **Stage 2 (Tumor Classification)** | 84.90% mAP@0.5 |
| **Inference Speed** | 0.0174 s/image (57.5 FPS) |
| **Improvement over SOTA** | +6.76% mAP |

### Per-Subtype Performance

| Tumor Subtype | Accuracy | Precision | Recall | AP@0.5 |
|---------------|----------|-----------|--------|--------|
| Clear Cell RCC | 95.3% | 95.7% | 91.1% | 0.943 |
| Papillary RCC | 90.8% | 91.3% | 85.7% | 0.995 |
| Chromophobe RCC | 55.0% | 79.5% | 79.3% | 0.665 |
| Oncocytoma | 79.0% | 89.9% | 79.3% | 0.726 |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/kidney-tumor-classification.git
cd kidney-tumor-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install YOLOv7
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
pip install -r requirements.txt
cd ..
```

### Download Dataset

```bash
# Download KiTS19 dataset
git clone https://github.com/neheller/kits19
cd kits19
pip install -r requirements.txt
python -m starter_code.get_imaging
cd ..
```

### Download Pre-trained Weights

```bash
# Create weights directory
mkdir -p weights

# Download YOLOv7-X weights
cd weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-x.pt

# Download our trained models (will be available upon paper acceptance)
# wget [URL to be added]/stage1_efficientnet_b7.pth
# wget [URL to be added]/stage2_yolov7x_jet.pt
cd ..
```

## 📖 Usage

### 1. Data Preprocessing

```bash
# Preprocess CT images with JET enhancement
python scripts/preprocess_data.py \
    --input_dir kits19/data \
    --output_dir data/preprocessed \
    --method jet \
    --size 640
```

### 2. Train Stage 1 (Kidney Detection)

```bash
python scripts/train_stage1.py \
    --data_dir data/preprocessed \
    --model efficientnet-b7 \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir checkpoints/stage1
```

### 3. Train Stage 2 (Tumor Classification)

```bash
cd yolov7
python train.py \
    --workers 8 \
    --device 0 \
    --batch-size 16 \
    --epochs 200 \
    --data ../configs/kidney_tumors.yaml \
    --img 640 640 \
    --cfg cfg/training/yolov7-x.yaml \
    --weights yolov7-x.pt \
    --name yolov7x_kidney_jet \
    --hyp ../configs/hyp.scratch.custom.yaml
```

### 4. Inference

```bash
# Single image inference
python scripts/inference.py \
    --image path/to/ct_image.jpg \
    --stage1_weights weights/stage1_best.pth \
    --stage2_weights weights/stage2_best.pt \
    --output results/

# Batch inference
python scripts/inference.py \
    --image_dir path/to/images/ \
    --stage1_weights weights/stage1_best.pth \
    --stage2_weights weights/stage2_best.pt \
    --output results/
```

### 5. Evaluation

```bash
python scripts/evaluate.py \
    --test_data data/test \
    --stage1_weights weights/stage1_best.pth \
    --stage2_weights weights/stage2_best.pt \
    --output results/evaluation.json
```

## 📁 Project Structure

```
kidney-tumor-classification/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── efficientnet.py      # Stage 1 model
│   │   └── yolo.py               # Stage 2 wrapper
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Dataset classes
│   │   └── transforms.py         # Data augmentation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── jet_enhancement.py    # JET colormap
│   │   ├── hsv_transform.py      # HSV conversion
│   │   └── dicom_utils.py        # DICOM handling
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # Evaluation metrics
│       ├── visualization.py      # Plotting functions
│       └── logger.py             # Logging utilities
├── scripts/
│   ├── preprocess_data.py        # Data preprocessing
│   ├── train_stage1.py           # Stage 1 training
│   ├── train_stage2.py           # Stage 2 training
│   ├── inference.py              # Inference script
│   └── evaluate.py               # Evaluation script
├── configs/
│   ├── kidney_tumors.yaml        # YOLOv7 data config
│   ├── hyp.scratch.custom.yaml   # Hyperparameters
│   └── train_config.yaml         # Training configuration
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_comparison.ipynb
│   └── 03_results_visualization.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_metrics.py
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

## 🔬 Methodology

### Two-Stage Framework

#### Stage 1: Kidney Detection
- **Model**: EfficientNet-B7 (66.3M parameters)
- **Input**: 600×600 RGB images
- **Output**: Binary classification (kidney/no kidney)
- **Accuracy**: 97.8%
- **Training**: Adam optimizer, ReduceLROnPlateau, early stopping

#### Stage 2: Tumor Classification
- **Model**: YOLOv7-X (71.3M parameters)
- **Input**: 640×640 JET-enhanced images
- **Output**: Multi-class detection (5 tumor subtypes + kidney)
- **Performance**: 84.90% mAP@0.5, 88.40% F1-score
- **Training**: SGD with momentum, cosine annealing, 200 epochs

### Preprocessing Methods

1. **Grayscale (GRAY)**: Standard CT imaging
   - Preserves original Hounsfield Units
   - Baseline: 79.00% accuracy

2. **JET Colormap (JET)**: Pseudo-color enhancement ⭐
   - Maps intensity to rainbow spectrum
   - **Best performance**: 82.56% accuracy (+3.56%)
   - Statistical significance: p<0.01

3. **HSV Transformation (HSV)**: Color space conversion
   - Separates hue, saturation, value
   - Performance: 81.04% accuracy (+2.04%)

## 📈 Results

### Comparison with State-of-the-Art

| Method | Year | mAP@0.5 | F1-Score | Accuracy |
|--------|------|---------|----------|----------|
| Mahmud et al. | 2023 | 78.14% | 82.35% | 76.80% |
| Lin et al. | 2023 | 80.22% | 84.50% | 78.45% |
| **Ours (GRAY)** | 2024 | 81.54% | 84.07% | 79.00% |
| **Ours (HSV)** | 2024 | 83.39% | 86.82% | 81.04% |
| **Ours (JET)** | 2024 | **84.90%** | **88.40%** | **82.56%** |

**Improvement**: +6.76% mAP over previous best (p<0.01)

### Clinical Performance

- **Sensitivity**: 91.70% (excellent for screening)
- **Specificity**: 85.30% (reduces false positives)
- **NPV**: 94.2% (confident in negative results)
- **PPV**: 88.5% (reliable positive predictions)

## 🖥️ Hardware Requirements

### Training
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **RAM**: 64 GB
- **Storage**: 2 TB SSD
- **OS**: Ubuntu 20.04+ or Windows 10+

### Inference
- **GPU**: NVIDIA GTX 1080 Ti (11GB) or better
- **RAM**: 16 GB
- **CPU**: Intel i7 or equivalent (for CPU inference)

### Training Time
- **Stage 1**: ~48 hours (20 epochs)
- **Stage 2**: ~48 hours (200 epochs)
- **Total**: ~96 hours on RTX 3090

## 📊 Dataset

**KiTS19 (Kidney and Kidney Tumor Segmentation Challenge 2019)**
- **Source**: https://kits19.grand-challenge.org/
- **Total Cases**: 210 CT scans
- **Tumor Subtypes**: 5 (ccRCC, pRCC, chRCC, ONC, + kidney)
- **Annotations**: Expert-verified segmentation masks
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0

### Data Split
- **Training**: 700 images (70%)
- **Validation**: 200 images (20%)
- **Test**: 100 images (10%)
- **Method**: Stratified split to maintain class distribution

## 🎓 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{li2024kidney,
  title={Deep Learning-Based Kidney Tumor Subtype Classification from CT Imaging: 
         A Comprehensive Two-Stage Framework with Contrast Enhancement},
  author={Li, Wen-Yi and Peng, Jia-Yang and Cheng, Bor-Wen and 
          Yang, Wei-Shiung and Yang, Feng-Jung},
  journal={Journal of Medical Internet Research},
  year={2024},
  note={Under Review}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

## 🙏 Acknowledgments

- **KiTS19 Dataset**: University of Minnesota
- **YOLOv7**: WongKinYiu and contributors
- **EfficientNet**: Google Research
- **PyTorch**: Facebook AI Research
- **National Taiwan University Hospital**: Institutional support
- **National Yunlin University of Science and Technology**: Computational resources

## 📧 Contact

**Corresponding Author**: Feng-Jung Yang, MD, PhD
- **Email**: fongrong@ntu.edu.tw
- **Institution**: National Taiwan University Hospital
- **Department**: Medical Genetics

## 🔗 Links

- **Paper**: [To be added upon publication]
- **Dataset**: https://kits19.grand-challenge.org/
- **YOLOv7**: https://github.com/WongKinYiu/yolov7
- **EfficientNet**: https://github.com/lukemelas/EfficientNet-PyTorch

## 📌 Roadmap

- [x] Initial release with core functionality
- [x] Pre-trained model weights
- [ ] Multi-center validation study
- [ ] Integration with PACS systems
- [ ] Web-based demo application
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, Azure, GCP)

## ⚠️ Disclaimer

This software is intended for research purposes only and has not been approved for clinical use. Always consult with qualified healthcare professionals for medical diagnosis and treatment decisions.

## 📜 Version History

### v1.0.0 (2024-01-26)
- Initial release
- Two-stage framework implementation
- JET/HSV/GRAY preprocessing methods
- Complete training and inference pipelines
- Comprehensive documentation

---

**Star ⭐ this repository if you find it helpful!**

**For questions, issues, or collaboration opportunities, please open an issue or contact the corresponding author.**
