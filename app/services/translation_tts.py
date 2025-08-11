from typing import Optional, Dict, Any
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
from logging import getLogger

logger = getLogger(__name__)

class TranslationTTSService:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'kn': 'Kannada',
            'te': 'Telugu'
        }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of the input text"""
        try:
            # Use Google Translator to detect language
            translator = GoogleTranslator(source='auto', target='en')
            detected_lang = translator.detect_language(text)
            return {
                "language": detected_lang,
                "confidence": 0.9,  # deep-translator doesn't provide confidence
                "language_name": self.supported_languages.get(detected_lang, detected_lang)
            }
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {"error": f"Language detection failed: {str(e)}"}
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = 'auto') -> Dict[str, Any]:
        """Translate text to target language"""
        try:
            if source_lang == 'auto':
                # Detect source language first
                detection = self.detect_language(text)
                if "error" not in detection:
                    source_lang = detection["language"]
                else:
                    source_lang = 'en'  # fallback to English
            
            # Use Google Translator
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_text = translator.translate(text)
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "source_language_name": self.supported_languages.get(source_lang, source_lang),
                "target_language_name": self.supported_languages.get(target_lang, target_lang),
                "pronunciation": None  # deep-translator doesn't provide pronunciation
            }
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {"error": f"Translation failed: {str(e)}"}
    
    def text_to_speech(self, text: str, language: str = 'en', slow: bool = False) -> Dict[str, Any]:
        """Convert text to speech and return base64 encoded audio"""
        try:
            # Create TTS object
            tts = gTTS(text=text, lang=language, slow=slow)
            
            # Use BytesIO to store audio in memory instead of saving to disk
            from io import BytesIO
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            
            # Get the audio data and encode to base64
            audio_buffer.seek(0)
            audio_data = audio_buffer.getvalue()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Close the buffer
            audio_buffer.close()
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "audio_format": "audio/mp3",
                "text": text,
                "language": language,
                "language_name": self.supported_languages.get(language, language)
            }
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return {"error": f"Text-to-speech failed: {str(e)}"}
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages
    
    def get_language_name(self, lang_code: str) -> str:
        """Get language name from language code"""
        return self.supported_languages.get(lang_code, lang_code)

# Create global instance
translation_tts_service = TranslationTTSService()
