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

## Core Module

```python
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as tf_keras_backend
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, LSTM, Bidirectional, Dense, Input, Reshape, Dropout, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from keras.layers import SpatialDropout2D

def Model1():
    inputs = Input(shape=(32, 128, 1))  

    # First CNN layer
    conv_1 = Conv2D(128, (5, 5), activation='relu', padding='same')(inputs)  
    batch_norm_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(batch_norm_1)
    dropout_1 = SpatialDropout2D(0.2)(pool_1)  

    # Second CNN layer
    conv_2a = Conv2D(256, (3, 3), activation='relu', padding='same')(dropout_1)  
    conv_2b = Conv2D(256, (5, 5), activation='relu', padding='same')(dropout_1)  
    concat_2 = Concatenate()([conv_2a, conv_2b])
    batch_norm_2 = BatchNormalization()(concat_2)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(batch_norm_2)
    dropout_2 = SpatialDropout2D(0.2)(pool_2)  

    # Third and fourth CNN layers
    conv_3a = Conv2D(512, (3, 3), activation='relu', padding='same')(dropout_2) 
    conv_3b = Conv2D(512, (5, 5), activation='relu', padding='same')(dropout_2)  
    concat_3 = Concatenate()([conv_3a, conv_3b])
    batch_norm_3 = BatchNormalization()(concat_3)

    conv_4 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_3)
    batch_norm_4 = BatchNormalization()(conv_4)
    pool_4 = MaxPool2D(pool_size=(2, 1))(batch_norm_4)
    dropout_4 = SpatialDropout2D(0.2)(pool_4)  

    # Fifth and sixth CNN layers
    conv_5a = Conv2D(1024, (3, 3), activation='relu', padding='same')(dropout_4)  
    conv_5b = Conv2D(1024, (5, 5), activation='relu', padding='same')(dropout_4)  
    concat_5 = Concatenate()([conv_5a, conv_5b])
    batch_norm_5 = BatchNormalization()(concat_5)

    conv_6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    dropout_6 = SpatialDropout2D(0.3)(pool_6)  

    # Final CNN layer
    conv_7 = Conv2D(2048, (2, 2), activation='relu')(dropout_6)  
    batch_norm_7 = BatchNormalization()(conv_7)

    # Reshape CNN output
    reshaped = Reshape((31, 2048))(batch_norm_7)  

    # LSTM layers
    blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(reshaped)  
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3))(blstm_1)  
    blstm_3 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.4))(blstm_2)  

    # Dense layer
    dense = Dense(1024, activation='relu')(blstm_3)  
    batch_norm_dense = BatchNormalization()(dense)
    dense_dropout = Dropout(0.4)(batch_norm_dense)  
    outputs = Dense(len(char_list) + 1, activation='softmax')(dense_dropout)

    act_model = Model(inputs, outputs)
    
    return act_model, outputs, inputs
```

---

## Results and Model Performance

### Training and Validation Metrics
![Training and Validation Metrics](https://github.com/user-attachments/assets/1f14b372-0436-45fd-b28b-f03ad0f21291)

The above chart demonstrates:
- Steady decreases in training and validation loss over epochs.
- Stable accuracy with minimal overfitting. The training accuracy reaches **97.64%**, while validation accuracy achieves **94.53%**, indicating effective generalization to unseen data.

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
