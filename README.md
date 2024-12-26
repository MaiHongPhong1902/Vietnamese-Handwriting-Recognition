# Vietnamese Handwriting Recognition Using CRNN

## Overview
This project explores a robust approach to Vietnamese handwriting recognition using Convolutional Recurrent Neural Networks (CRNN). The architecture combines CNNs, RNNs, and Connectionist Temporal Classification (CTC) to effectively handle spatial features and sequential dependencies in text images, making it suitable for recognizing handwritten Vietnamese text, including tonal marks and diacritics.

---

## Key Features
- **Model Architecture**: Combines CNN for feature extraction, RNN (LSTM/BLSTM) for sequence modeling, and CTC for decoding variable-length output sequences.
- **Target Languages**: Vietnamese and English handwriting recognition.
- **Applications**: Document digitization, data entry automation, and natural language processing.

---

## Project Components
### Theoretical Foundations
1. **CNN**: Extracts spatial features (e.g., edges, textures).
2. **RNN**: Captures sequential dependencies in character data.
3. **CTC**: Decodes sequences without requiring segmentation.

### Dataset
- **Machine-Generated Text**: Used in initial training phases to teach character structures.
- **Handwritten Text**: Includes datasets tailored for Vietnamese (diacritics and tonal features) and English.

### Preprocessing
- Image normalization, noise reduction, and segmentation.
- Tailored for the complexity of Vietnamese characters.

---

## Results and Model Performance

### Training and Validation Metrics
![Training and Validation Metrics](https://github.com/user-attachments/assets/1f14b372-0436-45fd-b28b-f03ad0f21291)

The above chart demonstrates:
- Steady decreases in training and validation loss over epochs.
- Stable accuracy with minimal overfitting. The training accuracy reaches **97.64%**, while validation accuracy achieves **94.53%**, indicating effective generalization to unseen data.

---

### Input Image
![Input Image](https://github.com/user-attachments/assets/43f1f6be-d0a7-483e-9c97-e8531a3a815c)

This image serves as the test input for evaluating the model's performance on real-world handwriting.

---

### Bounding Box
![Bounding Box](https://github.com/user-attachments/assets/6af93863-af43-4f8c-bc40-ed5890e7db42)

Bounding boxes are drawn around individual words, aiding the model in segmenting and recognizing text effectively.

---

### Output Results
#### Row 1
![Row 1](https://github.com/user-attachments/assets/64ed3dd8-04a9-4a22-a5df-163b38f0e056)
**Predicted Text**: Đây là hình ảnh để test mô hình

#### Row 2
![Row 2](https://github.com/user-attachments/assets/ef4d5bfc-0593-4c65-ba00-9037d7c73717)
**Predicted Text**: Nhằm kiểm tra độ chính xác của

#### Row 3
![Row 3](https://github.com/user-attachments/assets/090dcb43-6294-4331-908c-1a74ce4f2121)
**Predicted Text**: mô hình đã Train

#### Row 4
![Row 4](https://github.com/user-attachments/assets/aaf2fdfc-b143-4dcb-aa2f-a10afb5aebbf)
**Predicted Text**: Train by Mai Hồng Phong

---

## How to Use
1. **Setup Environment**:
   - Install dependencies (TensorFlow, Keras, OpenCV, etc.).
   - Prepare datasets (machine-generated and handwritten text samples).

2. **Train the Model**:
   - Use the provided architecture to train the model in two stages:
     - Stage 1: Machine-generated text.
     - Stage 2: Handwritten text.

3. **Test the Model**:
   - Evaluate performance on unseen datasets.
   - Validate using bounding box-based noise filtering.

4. **Optimize**:
   - Employ data augmentation, hyperparameter tuning, and regularization techniques for better results.

---

## Future Enhancements
- **Data Augmentation**: Increase dataset diversity through rotation, flipping, and shifting.
- **Ensemble Learning**: Combine predictions from multiple models for improved stability.
- **Transfer Learning**: Utilize pretrained models for quicker convergence.

---

## Acknowledgments
Special thanks to PhD. Nguyễn Ngô Lâm and Ho Chi Minh City University of Technology and Education for their guidance and support throughout the project.

---

## Contact
- **Author**: Mai Hồng Phong
- **Student Code**: 20119192
- **Email**: maihongphong.study@gmail.com
