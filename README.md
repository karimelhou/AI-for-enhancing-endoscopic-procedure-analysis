# AI for Enhancing Endoscopic Procedure Analysis

**UNIVERSITE ABDELMALEK ESSAADI**  
**FACULTÉ DES SCIENCES ET TECHNIQUES DE TANGER**  
**DÉPARTEMENT GÉNIE INFORMATIQUE**  

**Mémoire de Projet de Fin d'Études**  
**MASTER SCIENCES ET TECHNIQUES EN SYSTÈMES INFORMATIQUES ET MOBILES**  

## Project Title
**Advanced Image Processing and Computer Vision Techniques for Enhancing Endoscopic Procedure Analysis**

## Abstract

This project explores the development of advanced AI models to assist in the analysis of endoscopic images and videos. The goal is to enhance the accuracy of anomaly detection during endoscopic procedures by leveraging image processing, classification, and segmentation techniques. Key features of this project include:

- Preprocessing of endoscopic images.
- Classification and segmentation models for anomaly detection.
- Real-time feedback using a user-friendly interface for medical professionals.

## Repository Structure

This repository contains the following key components:

- **code/**: Contains the source code for the AI models and processing scripts.
- **documents/**: Contains the thesis report, presentations, and other related documents.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/AI-for-enhancing-endoscopic-procedure-analysis.git
    cd AI-for-enhancing-endoscopic-procedure-analysis
    ```

2. **Set up the environment**:
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the application**:
   - Start the Gradio interface for image classification and segmentation:
     ```bash
     python app.py
     ```

4. **Upload your images**:
   - Use the Gradio interface to upload endoscopic images and view the model’s predictions in real-time.

## Technologies Used

- **Programming Languages**: Python
- **Machine Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV
- **Web Interface**: Gradio, HTML, CSS
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Results

The models trained for this project have demonstrated significant improvements in detecting anomalies, especially polyps, in endoscopic procedures. Some key performance metrics:

- **Classification Accuracy**: 99% (EfficientNetB4)
- **Segmentation Dice Coefficient**: 0.91 (DeepLabv3+)

## Future Work

- Extension to continuous video stream analysis.
- Real-world deployment in clinical environments.
- Application of AI techniques to other medical imaging modalities.

## Contributors

- **Karim El Houmaini** - Master's Thesis Student
- **Pr. ZOUHAIR Abdelhamid** - Supervisor
- **KAMMAS Chaimaa** - External Supervisor

## Acknowledgments

I would like to extend my gratitude to the jury members, my supervisors, and the team at CHU Tangier for their invaluable support throughout this project.
