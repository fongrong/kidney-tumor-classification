# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-26

### Added
- Initial release
- Two-stage kidney tumor classification framework
- EfficientNet-B7 for kidney detection (Stage 1)
- YOLOv7-X for tumor classification (Stage 2)
- Three preprocessing methods (GRAY, JET, HSV)
- JET colormap enhancement achieving +3.56% accuracy
- Complete training and inference pipelines
- Comprehensive evaluation metrics
- Data augmentation strategies
- Documentation and examples
- Test suite
- GitHub Actions CI/CD

### Performance
- Stage 1 accuracy: 97.8%
- Stage 2 mAP@0.5: 84.90%
- Inference speed: 0.0174 s/image (57.5 FPS)
- State-of-the-art performance on KiTS19 dataset

### Documentation
- Comprehensive README
- Installation guide
- Usage examples
- API documentation
- Contributing guidelines
- Code of conduct

## [Unreleased]

### Planned
- Web-based demo application
- Multi-center validation study
- Mobile deployment (TensorFlow Lite)
- Docker containerization
- Cloud deployment guides
- Additional preprocessing methods
- Model compression and optimization
