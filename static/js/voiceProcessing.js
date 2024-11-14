/**
 * Clase para procesar y gestionar la grabación y análisis de voz
 */
class VoiceProcessor {
    constructor() {
        // Audio Context y componentes de audio
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        
        // Estado de grabación
        this.isRecording = false;
        this.animationFrame = null;
        this.voiceKey = null;
        this.stream = null;
        this.recognizedText = '';
        
        // Charts para visualización
        this.waveformChart = null;
        this.spectrumChart = null;
        this.waveformChartVerify = null;
        this.spectrumChartVerify = null;
        
        // Modo actual (encrypt/decrypt)
        this.currentMode = 'encrypt';
        
        // Constantes de configuración
        this.CONFIG = {
            sampleRate: 44100,
            fftSize: 2048,
            smoothingTimeConstant: 0.8,
            maxRecordingTime: 5000, // 5 segundos
            minAudioDuration: 0.5,  // 0.5 segundos
            silenceThreshold: 0.01,
            audioConstraints: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        };
    }

    async initializeCharts() {
        try {
            // Obtener contextos de los canvas
            const contexts = {
                waveform: document.getElementById('voice-waveform')?.getContext('2d'),
                spectrum: document.getElementById('voice-spectrum')?.getContext('2d'),
                waveformVerify: document.getElementById('voice-waveform-verify')?.getContext('2d'),
                spectrumVerify: document.getElementById('voice-spectrum-verify')?.getContext('2d')
            };

            // Inicializar gráficos si existen los contextos
            if (contexts.waveform) {
                this.waveformChart = this.createWaveformChart(contexts.waveform);
            }
            if (contexts.spectrum) {
                this.spectrumChart = this.createSpectrumChart(contexts.spectrum);
            }
            if (contexts.waveformVerify) {
                this.waveformChartVerify = this.createWaveformChart(contexts.waveformVerify);
            }
            if (contexts.spectrumVerify) {
                this.spectrumChartVerify = this.createSpectrumChart(contexts.spectrumVerify);
            }

            console.log('Charts initialized successfully');
        } catch (error) {
            console.error('Error initializing charts:', error);
            throw error;
        }
    }

    createWaveformChart(ctx) {
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(100).fill(''),
                datasets: [{
                    data: Array(100).fill(128),
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { min: 0, max: 255 }
                }
            }
        });
    }

    createSpectrumChart(ctx) {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array(100).fill(''),
                datasets: [{
                    data: Array(100).fill(0),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { beginAtZero: true }
                }
            }
        });
    }

    setupEventListeners() {
        // Configurar listeners para botones de grabación
        const buttons = {
            startRecording: document.getElementById('start-recording'),
            stopRecording: document.getElementById('stop-recording'),
            startVerify: document.getElementById('start-verify'),
            stopVerify: document.getElementById('stop-verify'),
            encryptFile: document.getElementById('encrypt-file'),
            decryptFile: document.getElementById('decrypt-file')
        };

        if (buttons.startRecording) {
            buttons.startRecording.addEventListener('click', () => this.startRecording('encrypt'));
        }
        if (buttons.stopRecording) {
            buttons.stopRecording.addEventListener('click', () => this.stopRecording('encrypt'));
        }
        if (buttons.startVerify) {
            buttons.startVerify.addEventListener('click', () => this.startRecording('decrypt'));
        }
        if (buttons.stopVerify) {
            buttons.stopVerify.addEventListener('click', () => this.stopRecording('decrypt'));
        }
        if (buttons.encryptFile) {
            buttons.encryptFile.addEventListener('click', () => this.encryptFile());
        }
        if (buttons.decryptFile) {
            buttons.decryptFile.addEventListener('click', () => this.decryptFile());
        }
    }

    async startRecording(mode = 'encrypt') {
        try {
            console.log('Starting recording in mode:', mode);
            this.currentMode = mode;
    
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: this.CONFIG.sampleRate
                });
                this.analyser = this.audioContext.createAnalyser();
                this.analyser.fftSize = this.CONFIG.fftSize;
            }
    
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
    
            // Configuración mejorada para la grabación
            const constraints = {
                audio: {
                    channelCount: 1,
                    sampleRate: this.CONFIG.sampleRate,
                    sampleSize: 16,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    volume: 1.0
                }
            };
    
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log('Audio stream created:', this.stream.getAudioTracks()[0].getSettings());
    
            const source = this.audioContext.createMediaStreamSource(this.stream);
            source.connect(this.analyser);
    
            const mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                throw new Error('El formato de audio no es soportado por este navegador');
            }
    
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: mimeType,
                audioBitsPerSecond: 128000
            });
    
            this.recordedChunks = [];
            this.isRecording = true;
    
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
    
            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.recordedChunks, { type: mimeType });
                await this.processAndSaveAudio(audioBlob, mode);
            };
    
            this.mediaRecorder.start();
            this.startVisualization();
            this.updateRecordingUI(true, mode);
            this.updateStatus('Grabando... Habla claramente', 'info', mode);
    
            // Agregar indicador visual de tiempo
            let timeLeft = 5;
            const timerElement = document.createElement('div');
            timerElement.className = 'mt-2 text-sm text-gray-600';
            const statusElement = document.getElementById(`${mode}-status`);
            if (statusElement) {
                statusElement.appendChild(timerElement);
            }
    
            const timer = setInterval(() => {
                timerElement.textContent = `Tiempo restante: ${timeLeft} segundos`;
                timeLeft--;
                if (timeLeft < 0) {
                    clearInterval(timer);
                }
            }, 1000);
    
            setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording(mode);
                }
            }, 5000);
    
        } catch (error) {
            console.error('Error starting recording:', error);
            this.updateStatus(
                `Error al acceder al micrófono: ${error.message}. Asegúrate de que el micrófono esté conectado y permitido.`,
                'error',
                mode
            );
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            this.isRecording = false;
        }
    }

    async processAndSaveAudio(audioBlob, mode) {
        try {
            console.log('Processing audio...', {
                mode: mode,
                blobSize: audioBlob.size,
                blobType: audioBlob.type
            });
    
            // Verificar calidad del audio
            try {
                await this.checkAudioQuality(audioBlob);
            } catch (qualityError) {
                this.updateStatus(qualityError.message, 'error', mode);
                return;
            }
    
            // Convertir a WAV
            const formData = new FormData();
            formData.append('audio', audioBlob);
    
            const response = await fetch('/convert_to_wav', {
                method: 'POST',
                body: formData
            });
    
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Error al convertir el audio');
            }
    
            // Guardar y procesar el WAV
            const wavBlob = await response.blob();
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            this.downloadFile(wavBlob, `voice-key-${timestamp}.wav`);
    
            // Procesar para obtener clave de voz
            const voiceFormData = new FormData();
            voiceFormData.append('audio', audioBlob);
    
            const voiceResponse = await fetch('/process_voice', {
                method: 'POST',
                body: voiceFormData
            });
    
            if (!voiceResponse.ok) {
                const errorData = await voiceResponse.json();
                throw new Error(errorData.error || 'Error al procesar la voz');
            }
    
            const data = await voiceResponse.json();
            
            if (data.voice_key) {
                this.voiceKey = data.voice_key;
                this.recognizedText = data.text || '';
                
                const actionButton = document.getElementById(
                    mode === 'encrypt' ? 'encrypt-file' : 'decrypt-file'
                );
                if (actionButton) {
                    actionButton.disabled = false;
                }
    
                this.updateStatus('Clave de voz generada y guardada como WAV', 'success', mode);
                
                const textElement = document.getElementById(
                    mode === 'encrypt' ? 'recognized-text' : 'verify-recognized-text'
                );
                if (textElement) {
                    textElement.textContent = this.recognizedText || 'No se reconoció texto';
                }
            }
    
        } catch (error) {
            console.error('Error processing audio:', error);
            this.updateStatus(`Error: ${error.message}`, 'error', mode);
            
            // Mostrar notificación más detallada al usuario
            const errorDetails = document.createElement('div');
            errorDetails.className = 'mt-2 text-sm text-gray-600';
            errorDetails.textContent = 'Sugerencias: Habla más cerca del micrófono, evita el ruido de fondo, y asegúrate de hablar durante al menos 1 segundo.';
            
            const statusElement = document.getElementById(`${mode}-status`);
            if (statusElement) {
                statusElement.appendChild(errorDetails);
            }
        }
    }

    async checkAudioQuality(audioBlob) {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob);
    
            console.log('Checking audio quality...', {
                blobSize: audioBlob.size,
                blobType: audioBlob.type
            });
    
            const response = await fetch('/check_audio', {
                method: 'POST',
                body: formData
            });
    
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Server error during audio check:', errorData);
                throw new Error(errorData.error || 'Error en la verificación del audio');
            }
    
            const diagnostics = await response.json();
            console.log('Audio diagnostics:', diagnostics);
            
            // Verificar problemas específicos
            if (!diagnostics.is_good_quality) {
                let errorMessage = 'Calidad de audio insuficiente:';
                if (diagnostics.issues && diagnostics.issues.length > 0) {
                    errorMessage += ' ' + diagnostics.issues.join(', ');
                }
                if (diagnostics.duration < 0.5) {
                    errorMessage += ' El audio es demasiado corto (mínimo 0.5 segundos).';
                }
                if (diagnostics.mean_amplitude < 0.01) {
                    errorMessage += ' El volumen es demasiado bajo.';
                }
                if (diagnostics.silence_ratio > 0.7) {
                    errorMessage += ' Demasiado silencio en la grabación.';
                }
                throw new Error(errorMessage);
            }
    
            return diagnostics;
    
        } catch (error) {
            console.error('Error checking audio quality:', error);
            // Reenviar el error con un mensaje más descriptivo
            throw new Error(`Error al verificar la calidad del audio: ${error.message}`);
        }
    }

    stopRecording(mode = 'encrypt') {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.stream.getTracks().forEach(track => track.stop());
        }

        this.isRecording = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }

        this.updateRecordingUI(false, mode);
        this.updateStatus('Procesando audio...', 'info', mode);
    }

    startVisualization() {
        const draw = () => {
            if (!this.isRecording) return;

            const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            const waveformArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            this.analyser.getByteTimeDomainData(waveformArray);
            this.analyser.getByteFrequencyData(dataArray);

            // Actualizar indicador de volumen
            const volume = dataArray.reduce((acc, val) => acc + val, 0) / dataArray.length;
            this.updateVolumeIndicator(volume);

            // Actualizar gráficos según el modo
            const charts = this.currentMode === 'encrypt' 
                ? [this.waveformChart, this.spectrumChart]
                : [this.waveformChartVerify, this.spectrumChartVerify];

            if (charts[0]) {
                charts[0].data.datasets[0].data = Array.from(waveformArray);
                charts[0].update('none');
            }
            if (charts[1]) {
                charts[1].data.datasets[0].data = Array.from(dataArray);
                charts[1].update('none');
            }

            this.animationFrame = requestAnimationFrame(draw);
        };

        draw();
    }

    updateVolumeIndicator(volume) {
        const indicator = document.getElementById(
            `voice-level-indicator${this.currentMode === 'decrypt' ? '-verify' : ''}`
        );
        
        if (indicator) {
            const percentage = Math.min(100, (volume / 128) * 100);
            indicator.style.width = `${percentage}%`;
            
            // Actualizar color según el volumen
            const colorClass = volume < 50 ? 'bg-yellow-500' 
                           : volume > 200 ? 'bg-red-500' 
                           : 'bg-blue-500';
                           
            ['bg-blue-500', 'bg-yellow-500', 'bg-red-500'].forEach(cls => {
                indicator.classList.remove(cls);
            });
            indicator.classList.add(colorClass);
        }
    }

    async encryptFile() {
        const fileInput = document.getElementById('file-to-encrypt');
        const nickname = document.getElementById('nickname')?.value;
    
        if (!fileInput?.files.length || !nickname || !this.voiceKey) {
            this.updateStatus('Por favor proporciona toda la información requerida', 'error', 'encrypt');
            return;
        }
    
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('voice_key', this.voiceKey);
        formData.append('nickname', nickname);
        formData.append('recognized_text', this.recognizedText);
    
        try {
            const response = await fetch('/encrypt', {
                method: 'POST',
                body: formData
            });
    
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Error durante el cifrado');
            }
    
            const data = await response.json();
    
            if (data.success) {
                this.updateStatus(`Archivo cifrado exitosamente. ID: ${data.file_id}`, 'success', 'encrypt');
                this.resetForm('encrypt');
                
                // Guardar ID en el portapapeles
                await navigator.clipboard.writeText(data.file_id);
                this.updateStatus(`ID copiado al portapapeles: ${data.file_id}`, 'success', 'encrypt');
            }
    
        } catch (error) {
            console.error('Error durante el cifrado:', error);
            this.updateStatus(`Error durante el cifrado: ${error.message}`, 'error', 'encrypt');
        }
    }

    async decryptFile() {
        try {
            const fileId = document.getElementById('file-id')?.value.trim();
            const voiceKeyFile = document.getElementById('voice-key-file')?.files[0];
    
            if (!fileId || !voiceKeyFile) {
                this.updateStatus('Por favor proporciona el ID del archivo y el archivo WAV', 'error', 'decrypt');
                return;
            }
    
            console.log('Procesando archivo WAV:', {
                name: voiceKeyFile.name,
                size: voiceKeyFile.size,
                type: voiceKeyFile.type,
                lastModified: new Date(voiceKeyFile.lastModified)
            });
    
            // Procesar archivo de voz
            const formData = new FormData();
            formData.append('audio', voiceKeyFile);
    
            this.updateStatus('Procesando archivo de voz...', 'info', 'decrypt');
    
            const voiceResponse = await fetch('/process_voice', {
                method: 'POST',
                body: formData
            });
    
            if (!voiceResponse.ok) {
                const errorData = await voiceResponse.json();
                console.error('Error en procesamiento de voz:', errorData);
                throw new Error(errorData.error || 'Error al procesar el archivo de voz');
            }
    
            const voiceData = await voiceResponse.json();
            console.log('Características extraídas:', voiceData);
    
            if (!voiceData.voice_key) {
                throw new Error('No se pudo generar la clave de voz');
            }
    
            // Preparar datos para descifrado
            const decryptFormData = new FormData();
            decryptFormData.append('voice_key', voiceData.voice_key);
            decryptFormData.append('recognized_text', voiceData.text || '');
            decryptFormData.append('features', JSON.stringify(voiceData.features || {}));
    
            this.updateStatus('Descifrando archivo...', 'info', 'decrypt');
    
            const response = await fetch(`/decrypt/${fileId}`, {
                method: 'POST',
                body: decryptFormData
            });
    
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Error en descifrado:', errorData);
                if (response.status === 401) {
                    const confidence = errorData.confidence !== undefined ? 
                        `(confianza: ${(errorData.confidence * 100).toFixed(1)}%)` : '';
                    throw new Error(`Verificación de voz fallida ${confidence}`);
                }
                throw new Error(errorData.error || 'Error durante el descifrado');
            }
    
            const blob = await response.blob();
            const filename = this.getFilenameFromResponse(response) || 'archivo_descifrado';
            
            // Descargar archivo
            this.downloadFile(blob, filename);
            this.updateStatus('Archivo descifrado exitosamente', 'success', 'decrypt');
    
            // Mostrar puntuación de similitud
            const confidence = response.headers.get('X-Voice-Confidence');
            if (confidence) {
                this.updateSimilarityScore(confidence);
            }
    
        } catch (error) {
            console.error('Error durante el descifrado:', error);
            this.updateStatus(`Error: ${error.message}`, 'error', 'decrypt');
            this.resetSimilarityScore();
        }
    }

    getFilenameFromResponse(response) {
        const disposition = response.headers.get('Content-Disposition');
        if (disposition) {
            const matches = /filename=([^;]+)/.exec(disposition);
            if (matches?.length > 1) {
                return matches[1].trim();
            }
        }
        return this.generateDefaultFilename();
    }

    generateDefaultFilename() {
        const timestamp = new Date().toISOString().replace(/[-:.]/g, '').slice(0, 14);
        return `archivo_descifrado_${timestamp}.bin`;
    }

    downloadFile(blob, filename) {
        try {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            
            document.body.appendChild(a);
            a.click();
            
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            console.log('Archivo descargado:', filename);
        } catch (error) {
            console.error('Error al descargar el archivo:', error);
            this.updateStatus('Error al descargar el archivo', 'error', this.currentMode);
        }
    }

    updateSimilarityScore(confidence) {
        const similarityElement = document.getElementById('similarity-score');
        if (similarityElement) {
            const confidencePercent = (parseFloat(confidence) * 100).toFixed(1);
            similarityElement.textContent = `Similitud de voz: ${confidencePercent}%`;
            similarityElement.className = parseFloat(confidence) >= 0.6 ? 
                'mt-2 text-sm text-green-600' : 
                'mt-2 text-sm text-red-600';
        }
    }

    resetSimilarityScore() {
        const similarityElement = document.getElementById('similarity-score');
        if (similarityElement) {
            similarityElement.textContent = '';
        }
    }

    resetForm(mode) {
        if (mode === 'encrypt') {
            const fileInput = document.getElementById('file-to-encrypt');
            const nicknameInput = document.getElementById('nickname');
            
            if (fileInput) fileInput.value = '';
            if (nicknameInput) nicknameInput.value = '';
            
            this.voiceKey = null;
            this.recognizedText = '';
            
            const encryptButton = document.getElementById('encrypt-file');
            if (encryptButton) encryptButton.disabled = true;
        } else {
            const fileIdInput = document.getElementById('file-id');
            const voiceKeyFileInput = document.getElementById('voice-key-file');
            
            if (fileIdInput) fileIdInput.value = '';
            if (voiceKeyFileInput) voiceKeyFileInput.value = '';
            
            this.resetSimilarityScore();
        }
    }

    updateRecordingUI(isRecording, mode) {
        const startBtn = document.getElementById(mode === 'encrypt' ? 'start-recording' : 'start-verify');
        const stopBtn = document.getElementById(mode === 'encrypt' ? 'stop-recording' : 'stop-verify');
        
        if (startBtn) startBtn.disabled = isRecording;
        if (stopBtn) stopBtn.disabled = !isRecording;
        
        // Actualizar estado visual
        const recordingIndicator = document.getElementById(`recording-indicator-${mode}`);
        if (recordingIndicator) {
            recordingIndicator.classList.toggle('hidden', !isRecording);
        }
    }

    updateStatus(message, type, mode) {
        const statusElement = document.getElementById(`${mode}-status`);
        if (!statusElement) return;
    
        statusElement.textContent = message;
        statusElement.className = 'mt-4 text-center';
        
        // Aplicar clases según el tipo de mensaje
        const colorClasses = {
            error: 'text-red-600',
            success: 'text-green-600',
            info: 'text-blue-600',
            warning: 'text-yellow-600'
        };
        
        statusElement.classList.add(colorClasses[type] || 'text-gray-600');
        console.log(`Estado actualizado (${type}):`, message);
    }
}

// Verificación de compatibilidad del navegador
function checkBrowserCompatibility() {
    const compatibilityIssues = [];

    if (!window.AudioContext && !window.webkitAudioContext) {
        compatibilityIssues.push('AudioContext no está soportado');
    }

    if (!window.MediaRecorder) {
        compatibilityIssues.push('MediaRecorder no está soportado');
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        compatibilityIssues.push('getUserMedia no está soportado');
    }

    return {
        isCompatible: compatibilityIssues.length === 0,
        issues: compatibilityIssues
    };
}

// Inicialización cuando se carga el documento
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const compatibility = checkBrowserCompatibility();
        if (!compatibility.isCompatible) {
            throw new Error(`Navegador no compatible: ${compatibility.issues.join(', ')}`);
        }

        const voiceProcessor = new VoiceProcessor();
        await voiceProcessor.initializeCharts();
        voiceProcessor.setupEventListeners();
        
        console.log('Voice processor initialized successfully');

        // Verificar permisos de micrófono
        try {
            const result = await navigator.permissions.query({ name: 'microphone' });
            console.log('Microphone permission status:', result.state);
            
            const permissionBanner = document.getElementById('audio-permission-banner');
            if (permissionBanner) {
                permissionBanner.style.display = result.state === 'prompt' ? 'block' : 'none';
            }
            
            if (result.state === 'denied') {
                throw new Error('Acceso al micrófono denegado. Por favor, habilita el acceso en la configuración de tu navegador.');
            }
        } catch (permissionError) {
            console.warn('Could not query microphone permissions:', permissionError);
        }

    } catch (error) {
        console.error('Failed to initialize voice processor:', error);
        const statusElements = document.querySelectorAll('[id$="-status"]');
        statusElements.forEach(element => {
            element.textContent = `Error: ${error.message}`;
            element.className = 'mt-4 text-center text-red-600';
        });

        const recordingButtons = document.querySelectorAll('button[id^="start-"], button[id^="stop-"]');
        recordingButtons.forEach(button => button.disabled = true);
    }
});