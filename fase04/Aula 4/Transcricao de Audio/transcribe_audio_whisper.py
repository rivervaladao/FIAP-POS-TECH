import speech_recognition as sr
import whisper
import tempfile
import os

def transcribe_audio_file_offline(audio_file_path, output_path, language="pt"):
    """
    Transcreve arquivo de áudio WAV usando OpenAI Whisper offline e
    salva o resultado no arquivo ./data/transcribe-whisper.txt.
    """
    # Cria reconhecedor do SpeechRecognition
    recognizer = sr.Recognizer()

    # Lê o arquivo de áudio usando sr.AudioFile
    with sr.AudioFile(audio_file_path) as source:
        print("Carregando arquivo de áudio:", audio_file_path)
        audio_data = recognizer.record(source)
        print("Leitura do arquivo concluída.")

    # Salva o áudio capturado em um arquivo WAV temporário
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data.get_wav_data())
        temp_filename = tmp_file.name

    try:
        # Carrega o modelo local do Whisper (ex.: 'small', 'base', ...)
        model = whisper.load_model("small")

        # Transcreve localmente, sem utilizar APIs externas
        result = model.transcribe(temp_filename, fp16=False, language=language)

        print("Transcrição concluída. Texto gerado:")
        print(result["text"])

        # Escreve o texto em ./data/transcribe-whisper.txt
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Cria a pasta se não existir
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(result["text"])
        print(f"Transcrição salva em: {output_path}")

    finally:
        # Remove o arquivo temporário
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(script_dir, '../../data', 'audio1.wav')
    text_output_path = os.path.join(script_dir,'../../data', 'transcricao1.txt')
    transcribe_audio_file_offline(audio_path, text_output_path, language="pt")

if __name__ == "__main__":
    caminho_audio = "meu_audio.wav"  # Arquivo .wav de entrada
