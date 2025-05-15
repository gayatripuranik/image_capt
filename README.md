# Multilingual Image Caption Generator with Webcam and Audio

## Project Description
This project is a web application that generates descriptive captions for images and converts them to speech in multiple languages. Users can upload images or capture them through their webcam, and the application will generate a caption, translate it to the selected language, and provide audio playback of the translated caption.

## Objective
To create an accessible, user-friendly application that can describe images in multiple languages with audio support, making visual content more accessible across language barriers.

## Key Features
- **Image Input Options**: 
  - Upload images from your device
  - Capture images in real-time using your webcam
  
- **Caption Generation**: 
  - Automatic caption generation using state-of-the-art image captioning model
  - Clear, descriptive captions that explain image content

- **Multilingual Support**: 
  - Translation to multiple languages including Spanish, French, German, Chinese, Hindi, Japanese, Russian, and Arabic
  - Original English caption displayed alongside translation

- **Text-to-Speech Functionality**: 
  - Automatic generation of audio from the translated caption
  - Immediate playback option within the application

- **User-Friendly Interface**: 
  - Tab-based navigation between upload and webcam options
  - Drag-and-drop image upload support
  - Real-time image preview
  - Responsive design for use on various devices

## Technologies Used

### Backend
- **Flask**: Python web framework for building the application backend
- **Hugging Face Transformers**: Used for image captioning (BLIP model) and translation (MarianMT models)
- **PyTorch**: Deep learning framework powering the AI models
- **gTTS (Google Text-to-Speech)**: For converting text to speech audio
- **PIL (Python Imaging Library)**: For image processing

### Frontend
- **HTML/CSS**: For structuring and styling the web interface
- **JavaScript**: For handling user interactions and AJAX requests
- **WebRTC API**: For webcam integration

### Models
- **BLIP (Bootstrapping Language-Image Pre-training)**: For image captioning
- **MarianMT**: For neural machine translation between languages

## Installation

```bash
# Clone the repository
git clone https://github.com/gayatripuranik/image_capt.git
cd image_capt

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required packages
pip install flask torch torchvision transformers Pillow gtts

# Run the application
python app.py
