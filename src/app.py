import subprocess
from flask import Flask, render_template, request, jsonify, send_file, make_response
import librosa
import numpy as np
from voice.analyzer import AudioFFTAnalyzer
from voice.verification import VoiceVerification
from crypto.cipher import create_encryption_key, encrypt_file, decrypt_file
from database.models import Database
from utils.audio import (
    verify_audio_quality, 
    convert_audio_format,
    load_and_normalize_audio,
    calculate_audio_features
)
import tempfile
import os
import uuid
import base64
from datetime import datetime, time
import json
from vosk import Model, KaldiRecognizer

app = Flask(__name__, 
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates')),
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
)

app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10MB max
    UPLOAD_FOLDER=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads')),
    DATABASE=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'instance', 'voice_encrypted_files.db')),
    VOSK_MODEL_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vosk-model-small-es-0.42')),
    ALLOWED_EXTENSIONS={'webm', 'wav', 'mp3', 'ogg'}
)


# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['DATABASE']), exist_ok=True)

# Initialize components
db = Database(app.config['DATABASE'])
analyzer = AudioFFTAnalyzer()
voice_verifier = VoiceVerification()

# Initialize Vosk model
try:
    model = Model(app.config['VOSK_MODEL_PATH'])
    print("Vosk model initialized successfully")
except Exception as e:
    print(f"Error initializing Vosk model: {str(e)}")
    print("The application will continue without speech recognition capabilities.")
    model = None
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)    
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_voice', methods=['POST'])
def process_voice_data():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    temp_paths = []
    
    try:
        frequency_bands = calculate_audio_features(audio_data, sample_rate)
        if isinstance(frequency_bands, np.ndarray):
            frequency_bands = frequency_bands.tolist()  # Convierte a lista solo si es ndarray

        # Create unique temp files
        temp_id = str(uuid.uuid4())
        original_path = os.path.join(tempfile.gettempdir(), f"original_{temp_id}")
        wav_path = os.path.join(tempfile.gettempdir(), f"converted_{temp_id}.wav")
        temp_paths.extend([original_path, wav_path])
        
        # Save original file
        audio_file.save(original_path)
        
        # Convert to WAV if needed
        if audio_file.content_type != 'audio/wav':
            wav_path = convert_audio_format(original_path)
            if not wav_path:
                return jsonify({'error': 'Audio conversion failed'}), 500
        
        # Load and analyze audio
        audio_data, sample_rate = load_and_normalize_audio(wav_path)
        
        # Verify audio quality
        quality = verify_audio_quality(audio_data, sample_rate)
        if not quality['is_good_quality']:
            return jsonify({
                'error': 'Poor audio quality',
                'issues': quality['issues']
            }), 400
        
        # Extract features
        features = analyzer.analyze_signal(audio_data)
        
        # Recognize speech if model is available
        recognized_text = ""
        if model is not None:
            try:
                rec = KaldiRecognizer(model, sample_rate)
                rec.SetWords(True)
                
                # Process audio in chunks
                chunk_size = int(sample_rate * 0.2)
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    if len(chunk) > 0:
                        rec.AcceptWaveform(chunk.tobytes())
                
                result = json.loads(rec.FinalResult())
                recognized_text = result.get('text', '')
                
            except Exception as e:
                print(f"Speech recognition error: {e}")
        
        # Generate voice signature
        voice_key = analyzer.get_voice_signature(features)
        
        mfcc = list(features.spectral_contrast[:13])  # Si ya es una lista, no hace falta .tolist()
        
        response_data = {
            'voice_key': voice_key,
            'text': recognized_text,
            'features': {
                'pitch': float(features.spectral_centroid),
                'frequency_bands': frequency_bands,
                'mfcc': mfcc
            }
        }

        return jsonify(response_data)
    
    except Exception as e:
        # Crear la respuesta de error correctamente
        response = jsonify({'error': str(e)})
        response.status_code = 500
        return response
    
    finally:
        # Cleanup temp files
        for path in temp_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {path}: {e}")

@app.route('/encrypt', methods=['POST'])
def encrypt_route():
    if 'file' not in request.files or 'voice_key' not in request.form:
        return jsonify({'error': 'Missing file or voice key'}), 400
    
    try:
        file = request.files['file']
        voice_key = request.form['voice_key']
        recognized_text = request.form.get('recognized_text', '')
        
        file_data = file.read()
        if not file_data:
            return jsonify({'error': 'Empty file'}), 400
        
        # Create encryption key and encrypt file
        key, salt = create_encryption_key(voice_key)
        encrypted_data = encrypt_file(file_data, key)
        
        # Generate unique file ID
        file_id = base64.urlsafe_b64encode(os.urandom(16)).decode()
        
        # Save to database
        success = db.save_encrypted_file(
            file_id=file_id,
            encrypted_data=encrypted_data,
            salt=salt,
            voice_key=voice_key,
            recognized_text=recognized_text,
            features=request.form.get('features', '{}')
        )
        
        if not success:
            return jsonify({'error': 'Database error'}), 500
            
        return jsonify({
            'success': True,
            'file_id': file_id
        })
        
    except Exception as e:
        print(f"Encryption error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/decrypt/<file_id>', methods=['POST'])
def decrypt_route(file_id):
    if not file_id:
        return jsonify({'error': 'Missing file ID'}), 400
    
    voice_key = request.form.get('voice_key')
    current_text = request.form.get('recognized_text', '')
    features = request.form.get('features', '{}')
    
    if not voice_key:
        return jsonify({'error': 'Missing voice key'}), 400
        
    try:
        # Parse voice features
        features = json.loads(features)
        
        # Get file from database
        file_data = db.get_encrypted_file(file_id)
        if not file_data:
            return jsonify({'error': 'File not found'}), 404
        
        # Verify voice
        is_valid, confidence = voice_verifier.verify_voice(
            features1=file_data.get('features', {}),
            features2=features,
            text1=file_data.get('recognized_text', ''),
            text2=current_text
        )
        
        if not is_valid:
            return jsonify({
                'error': 'Voice verification failed',
                'confidence': confidence
            }), 401
        
        # Decrypt file
        try:
            key, _ = create_encryption_key(voice_key, file_data['salt'])
            decrypted_data = decrypt_file(file_data['encrypted_data'], key)
            
            if not decrypted_data:
                raise ValueError("Decryption resulted in empty data")
            
            # Format filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'decrypted_file_{timestamp}'
            
            # Create response with decrypted file
            response = make_response(decrypted_data)
            response.headers.update({
                'Content-Type': 'application/octet-stream',
                'Content-Disposition': f'attachment; filename={filename}',
                'X-Voice-Confidence': str(confidence)
            })
            
            return response
            
        except Exception as e:
            return jsonify({
                'error': f'Decryption failed: {str(e)}',
                'confidence': confidence
            }), 401
            
    except Exception as e:
        print(f"Decryption error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert_to_wav', methods=['POST'])
def convert_to_wav():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    audio_file = request.files['audio']
    temp_paths = []
    
    try:
        # Create unique temp files
        temp_id = str(uuid.uuid4())
        original_path = os.path.join(tempfile.gettempdir(), f"original_{temp_id}.webm")
        wav_path = os.path.join(tempfile.gettempdir(), f"converted_{temp_id}.wav")
        temp_paths.extend([original_path, wav_path])
        
        # Save original file
        audio_file.save(original_path)
        
        # Convert to WAV
        converted_path = convert_audio_format(original_path)
        if not converted_path:
            return jsonify({'error': 'Audio conversion failed'}), 500
        
        # Read and return WAV file
        with open(converted_path, 'rb') as wav_file:
            response = make_response(wav_file.read())
            response.headers['Content-Type'] = 'audio/wav'
            response.headers['Content-Disposition'] = f'attachment; filename=voice_key_{temp_id}.wav'
            return response
            
    except Exception as e:
        print(f"Conversion error: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup temp files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_path}: {e}")

@app.route('/check_audio', methods=['POST'])
def check_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    try:
        audio_file = request.files['audio']
        temp_path = None
        
        try:
            # Guardar temporalmente el archivo
            temp_filename = f'temp_check_{uuid.uuid4()}.webm'
            temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
            audio_file.save(temp_path)
            
            # Convertir audio usando ffmpeg
            wav_path = os.path.join(tempfile.gettempdir(), f'temp_check_{uuid.uuid4()}.wav')
            try:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', temp_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '1',
                    wav_path
                ], check=True, capture_output=True)
                
                # Cargar y analizar el audio
                audio_data, sample_rate = librosa.load(wav_path, sr=44100, mono=True)
                
                # Calcular métricas detalladas
                diagnostics = {
                    'duration': float(librosa.get_duration(y=audio_data, sr=sample_rate)),
                    'max_amplitude': float(np.max(np.abs(audio_data))),
                    'mean_amplitude': float(np.mean(np.abs(audio_data))),
                    'rms_energy': float(np.sqrt(np.mean(audio_data**2))),
                    'silence_ratio': float(np.mean(np.abs(audio_data) < 0.01)),
                    'is_good_quality': True,
                    'issues': []
                }
                
                # Verificar cada aspecto de la calidad
                if diagnostics['duration'] < 0.5:
                    diagnostics['issues'].append('Audio demasiado corto (mínimo 0.5 segundos)')
                    diagnostics['is_good_quality'] = False
                
                if diagnostics['mean_amplitude'] < 0.001:
                    diagnostics['issues'].append('Volumen muy bajo')
                    diagnostics['is_good_quality'] = False
                
                if diagnostics['silence_ratio'] > 0.5:
                    diagnostics['issues'].append('Demasiado silencio')
                    diagnostics['is_good_quality'] = False
                
                if diagnostics['max_amplitude'] > 0.95:
                    diagnostics['issues'].append('Audio saturado')
                    diagnostics['is_good_quality'] = False

                print("Audio diagnostics:", diagnostics)
                return jsonify(diagnostics)
                
            finally:
                # Limpiar archivos temporales
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                    
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file: {e}")
                    
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return jsonify({'error': 'Error converting audio format'}), 500
    except Exception as e:
        print(f"Error checking audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_temp_files():
    """Clean up old temporary files"""
    temp_dir = tempfile.gettempdir()
    try:
        for filename in os.listdir(temp_dir):
            if filename.startswith(('temp_', 'original_', 'converted_')):
                filepath = os.path.join(temp_dir, filename)
                try:
                    if os.path.getmtime(filepath) < time.time() - 3600:  # Older than 1 hour
                        os.remove(filepath)
                except Exception as e:
                    print(f"Warning: Could not remove old temp file {filepath}: {e}")
    except Exception as e:
        print(f"Warning: Error during temp file cleanup: {e}")

# Initialize the application
def init_app():
    # Initialize database
    db.init_db()
    
    # Schedule cleanup task
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(cleanup_temp_files, 'interval', hours=1)
    scheduler.start()

if __name__ == '__main__':
    init_app()
    app.run(debug=True)