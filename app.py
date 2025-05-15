from flask import Flask, render_template, request, jsonify, send_file, Response
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from PIL import Image
import torch
from gtts import gTTS
import uuid
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

LANGUAGES = {
    'en': {'name': 'English', 'model': None, 'tts_code': 'en'},
    'es': {'name': 'Spanish', 'model': 'Helsinki-NLP/opus-mt-en-es', 'tts_code': 'es'},
    'fr': {'name': 'French', 'model': 'Helsinki-NLP/opus-mt-en-fr', 'tts_code': 'fr'},
    'de': {'name': 'German', 'model': 'Helsinki-NLP/opus-mt-en-de', 'tts_code': 'de'},
    'zh': {'name': 'Chinese', 'model': 'Helsinki-NLP/opus-mt-en-zh', 'tts_code': 'zh-CN'},
    'hi': {'name': 'Hindi', 'model': 'Helsinki-NLP/opus-mt-en-hi', 'tts_code': 'hi'},
    'ja': {'name': 'Japanese', 'model': 'Helsinki-NLP/opus-mt-en-jap', 'tts_code': 'ja'},
    'ru': {'name': 'Russian', 'model': 'Helsinki-NLP/opus-mt-en-ru', 'tts_code': 'ru'},
    'ar': {'name': 'Arabic', 'model': 'Helsinki-NLP/opus-mt-en-ar', 'tts_code': 'ar'},
}

translation_models = {}
translation_tokenizers = {}


def load_translation_model(lang_code):
    """Load translation model on demand"""
    if lang_code == 'en' or lang_code not in LANGUAGES:
        return None, None

    if lang_code not in translation_models:
        model_name = LANGUAGES[lang_code]['model']
        try:
            translation_models[lang_code] = MarianMTModel.from_pretrained(model_name)
            translation_tokenizers[lang_code] = MarianTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model for {lang_code}: {str(e)}")
            return None, None

    return translation_models[lang_code], translation_tokenizers[lang_code]


def translate_text(text, target_lang):
    """Translate text to target language"""
    if target_lang == 'en':
        return text

    model, tokenizer = load_translation_model(target_lang)
    if model is None or tokenizer is None:
        print(f"Translation model for {target_lang} not available, returning original text")
        return text

    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            translated = model.generate(**inputs)

        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        print(f"Translation error for {target_lang}: {str(e)}")
        return text


@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files and 'webcam_image' not in request.json:
        return jsonify({'error': 'No image uploaded'}), 400

    target_lang = request.form.get('language', 'en') if 'image' in request.files else request.json.get('language', 'en')

    unique_id = str(uuid.uuid4())
    image_filename = unique_id + '.jpg'
    audio_filename = unique_id + '.mp3'

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)

    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        image_file.save(image_path)
    else:  # Webcam image
        img_data = request.json['webcam_image'].split(',')[1]
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(img_data))

    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption_en = processor.decode(out[0], skip_special_tokens=True)

        caption_translated = translate_text(caption_en, target_lang)

        tts = gTTS(text=caption_translated, lang=LANGUAGES[target_lang]['tts_code'])
        tts.save(audio_path)

        return jsonify({
            'success': True,
            'caption_original': caption_en,
            'caption': caption_translated,
            'language': LANGUAGES[target_lang]['name'],
            'image_url': '/' + image_path,
            'audio_url': '/' + audio_path
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)