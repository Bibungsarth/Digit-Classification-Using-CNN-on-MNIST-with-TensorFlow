# Digit-Classification-Using-CNN-on-MNIST-with-TensorFlow
This project implements a handwritten digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. However, the primary focus is on exploring the security vulnerabilities of AI models by simulating adversarial attacks through image distortion techniques.The project demonstrates how encrypted traffic can be analyzed for cybersecurity purposes without decrypting sensitive information, preserving user privacy. The implemented CNN is trained using a labeled dataset of encrypted network traffic and evaluates the robustness of the classification against potential adversarial conditions, such as noise or distorted input.

## Key Features

### Data Preprocessing
- Prepares encrypted network traffic data for CNN input.
- Converts network packets or flows into grayscale image representations.
- Normalizes image data to ensure uniform input for the CNN.

### Model Development
- Implements a TensorFlow-based Convolutional Neural Network (CNN) to process and classify encrypted traffic.
- Model architecture includes:
  - Convolutional layers to extract features.
  - Pooling layers to reduce dimensionality.
  - Dense layers for final classification into malicious vs. normal traffic.

### Visualization
- Visualizes sample inputs to understand the data fed into the CNN.
- Evaluates the impact of adversarial distortions (e.g., noise, rotations) on classification performance.

### Adversarial Testing
- Simulates adversarial conditions by introducing distortions into input data.
- Analyzes model robustness and identifies vulnerabilities.

### Performance Evaluation
- Measures accuracy, precision, recall, and F1-score on test data.
- Assesses model robustness under clean and adversarial inputs.

## Steps to Execute

### Prerequisites
Ensure the following are installed in Google Colab:
- Python 3.8 or above
- TensorFlow
- NumPy
- Matplotlib

### Run the Code
1. **Download the Python files from the GitHub repository.**
2. **Load the files into your Google Colab environment.**
3. **Run the Python files in Colab.**

## Project Workflow

1. **Data Collection and Labeling**
   - Collect encrypted traffic samples using tools like Wireshark or tcpdump.
   - Label traffic based on its behavior (malicious or normal).

2. **Image Conversion**
   - Convert traffic data into images:
     - Use flow visualization techniques (e.g., byte distributions or flow graphs).
     - Normalize pixel values for CNN input.

3. **Model Training**
   - Train the CNN on the converted image dataset.
   - Optimize model performance using the Adam optimizer.

4. **Adversarial Testing**
   - Test model robustness by introducing distortions:
     - Add noise, rotate images, or apply blurring.
   - Analyze performance degradation to assess vulnerabilities.

## Model Architecture
The CNN model is structured as follows:
- **Input Layer**: Processes image representations of network traffic.
- **Convolutional Layers**: Extract spatial features from traffic images.
- **Pooling Layers**: Reduce feature map dimensions to prevent overfitting.
- **Fully Connected Layers**: Perform final classification into malicious or normal categories.
- **Output Layer**: Uses a softmax activation function to output probabilities.

## Results
The model achieves:
- High accuracy on clean test data.
- Highlights potential vulnerabilities under adversarial conditions (e.g., noise, distortion).

## Future Enhancements
- Extend to multi-class classification (e.g., different types of attacks).
- Implement techniques to improve robustness against adversarial inputs.
- Use additional datasets for better generalization.

## Technologies Used
- **TensorFlow**: For building and training the CNN.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **Network Traffic Visualization Tools**: For converting traffic into images.

## Acknowledgments
- The MNIST dataset was used as a proof-of-concept before applying encrypted traffic data.
- Research papers on flow-based traffic analysis inspired the methodology.

## References
1. Kommineni, S., Rao Pulakonti, S. M., Narayana Anudeep, K., Srinivas, K., and Sri Lakshmi Poojitha, K., “Adversarial Defense for MNIST: Investigating Adversarial Training and FGSM,” Proceedings of the International Conference on Machine Learning (ICML), 2021.
2. Niu, A., Zhang, K., Zhang, C., Zhang, C., Kweon, I. S., Yoo, C. D., and Zhang, Y., “Fast Adversarial Training with Noise Augmentation: A Unified Perspective on RandStart and GradAlign,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 6, pp. 1920-1931, Jun. 2020.
3. Rosenfeld, E., Winston, E., Ravikumar, P., and Kolter, J. Z., “Certified Robustness to Label-Flipping Attacks via Randomized Smoothing,” Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.
