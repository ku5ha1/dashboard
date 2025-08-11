from typing import Optional, Dict, Any
from deep_translator import GoogleTranslator
from elevenlabs import ElevenLabs
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
        """Convert text to speech using ElevenLabs and return base64 encoded audio"""
        try:
            # Get API key and voice ID from environment
            import os
            api_key = os.getenv("ELEVENLABS_API_KEY")
            voice_id = os.getenv("ELEVENLABS_VOICE_ID")
            
            if not api_key:
                return {"error": "ElevenLabs API key not configured"}
            
            if not voice_id:
                return {"error": "ElevenLabs Voice ID not configured"}
            
            # Create ElevenLabs client
            client = ElevenLabs(api_key=api_key)
            
            # Generate audio using ElevenLabs
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                output_format="mp3_44100_128",  # MP3 format for better compatibility
                text=text,
                model_id="eleven_multilingual_v2"  # Best for multiple languages
            )
            
            # Convert generator to bytes if needed
            if hasattr(audio, '__iter__') and not isinstance(audio, bytes):
                # If it's a generator/iterator, convert to bytes
                audio_bytes = b''.join(audio)
            else:
                audio_bytes = audio
            
            # Convert audio to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "audio_format": "audio/mp3",  # Changed back to MP3 for better compatibility
                "text": text,
                "language": language,
                "language_name": self.supported_languages.get(language, language)
            }
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            return {"error": f"Text-to-speech failed: {str(e)}"}
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages
    
    def get_language_name(self, lang_code: str) -> str:
        """Get language name from language code"""
        return self.supported_languages.get(lang_code, lang_code)

# Create global instance
translation_tts_service = TranslationTTSService()
