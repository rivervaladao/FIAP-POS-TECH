from moviepy import VideoFileClip
import os
import tempfile
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence
import yt_dlp
import urllib.parse
import asyncio
import whisper

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configura o cliente OpenAI para modo online
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configura o modelo Whisper para modo offline
model = "base"

def download_youtube_video(url, output_path):
    """Baixa um vídeo do YouTube usando yt-dlp e retorna o caminho do arquivo"""
    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'quiet': False,
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Baixando de: {url}")
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            print("Download do YouTube concluído!")
            return downloaded_file
    except Exception as e:
        raise Exception(f"Erro ao baixar vídeo do YouTube: {str(e)}")

def extract_audio_from_video(video_path, audio_path, start_time=None, end_time=None):
    """Extrai o áudio de um vídeo e salva no caminho especificado"""
    video = VideoFileClip(video_path)
    if start_time is not None or end_time is not None:
        start_time = start_time if start_time is not None else 0
        end_time = end_time if end_time is not None else video.duration
        video = video.subclip(start_time, end_time)
        print(f"Processando vídeo de {start_time}s a {end_time}s")
    with tqdm(total=100, desc="Extraindo áudio") as pbar:
        video.audio.write_audiofile(audio_path)
        pbar.update(100)

def split_audio_with_silence(audio_path, segment_duration=600, silence_thresh=-40, min_silence_len=500):
    """Divide o áudio em segmentos, ignorando trechos silenciosos, com barra de progresso"""
    audio = AudioSegment.from_wav(audio_path)
    total_duration_ms = len(audio)
    
    with tqdm(total=2, desc="Detectando silêncio", unit="passo") as pbar:
        pbar.update(1)
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200
        )
        pbar.update(1)
    
    result_chunks = []
    current_time = 0
    total_segments = 0
    for chunk in chunks:
        chunk_duration = len(chunk) / 1000
        total_segments += int(chunk_duration // segment_duration) + (1 if chunk_duration % segment_duration > 0 else 0)
    
    with tqdm(total=total_segments, desc="Dividindo segmentos", unit="segmento") as pbar:
        for chunk in chunks:
            chunk_duration = len(chunk) / 1000
            start_time = current_time
            
            while chunk_duration > segment_duration:
                segment = chunk[:segment_duration * 1000]
                segment_path = f"{audio_path}_segment_{int(start_time * 1000)}.wav"
                segment.export(segment_path, format="wav")
                result_chunks.append((segment_path, start_time, start_time + segment_duration))
                chunk = chunk[segment_duration * 1000:]
                start_time += segment_duration
                chunk_duration -= segment_duration
                pbar.update(1)
            
            if chunk_duration > 0:
                segment_path = f"{audio_path}_segment_{int(start_time * 1000)}.wav"
                chunk.export(segment_path, format="wav")
                result_chunks.append((segment_path, start_time, start_time + chunk_duration))
                pbar.update(1)
            
            current_time = start_time + chunk_duration
    
    print(f"Segmentos com som detectados: {len(result_chunks)}")
    return result_chunks

def transcribe_segment_online(segment_path):
    """Transcreve um segmento de áudio usando a API Whisper online (síncrono)"""
    try:
        with open(segment_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="pt"
            )
        return transcription.text
    except Exception as e:
        print(f"Erro ao transcrever segmento {segment_path} (online): {str(e)}")
        return None

def transcribe_segment_offline(segment_path, model):
    """Transcreve um segmento de áudio usando o modelo Whisper offline"""
    try:
        model = whisper.load_model(model)
        transcription = model.transcribe(segment_path)["text"]
        return transcription
    except Exception as e:
        print(f"Erro ao transcrever segmento {segment_path} (offline): {str(e)}")
        return None

async def transcribe_audio_to_text(audio_path, text_output_path, video_url=None, with_timestamp=True, segment_duration=600, mode="online", silence_thresh=-40, min_silence_len=500, model_name="base"):
    """Transcreve o áudio para texto usando a API Whisper ou modelo local, ignorando silêncio"""
    chunks = split_audio_with_silence(audio_path, segment_duration=segment_duration, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
    
    full_text = ""
    if mode == "online":
        with tqdm(total=len(chunks), desc="Transcrevendo (online)") as pbar:
            for chunk_path, start_time, end_time in chunks:
                chunk_text = transcribe_segment_online(chunk_path)
                if chunk_text:
                    if with_timestamp:
                        if video_url:
                            video_id = extract_video_id(video_url)
                            link = generate_timestamp_link(video_id, start_time)
                            full_text += f"[{link}]\n{chunk_text}\n\n"
                        else:
                            time_str = f"{format_time(start_time)} - {format_time(end_time)}"
                            full_text += f"[{time_str}]\n{chunk_text}\n\n"
                    else:
                        full_text += chunk_text + "\n\n"
                pbar.update(1)
    else:  # mode == "offline"
        with tqdm(total=len(chunks), desc="Transcrevendo (offline)") as pbar:
            for chunk_path, start_time, end_time in chunks:
                print(f"chunk_path: {chunk_path}")
                chunk_text = transcribe_segment_offline(chunk_path, model)
                if chunk_text:
                    if with_timestamp:
                        if video_url:
                            video_id = extract_video_id(video_url)
                            link = generate_timestamp_link(video_id, start_time)
                            full_text += f"[{link}]\n{chunk_text}\n\n"
                        else:
                            time_str = f"{format_time(start_time)} - {format_time(end_time)}"
                            full_text += f"[{time_str}]\n{chunk_text}\n\n"
                    else:
                        full_text += chunk_text + "\n\n"
                pbar.update(1)
    
    with open(text_output_path, 'w', encoding='utf-8') as file:
        file.write(full_text.strip())
    
    for chunk_path, _, _ in chunks:
        if os.path.exists(chunk_path):
            os.unlink(chunk_path)

def extract_video_id(url):
    """Extrai o ID do vídeo de uma URL do YouTube"""
    query = urllib.parse.urlparse(url).query
    return urllib.parse.parse_qs(query)['v'][0]

def generate_timestamp_link(video_id, start_time):
    """Gera um link do YouTube com timestamp"""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"

def format_time(seconds):
    """Formata o tempo em minutos:segundos"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def main():
    parser = argparse.ArgumentParser(description='Transcreve áudio de vídeo usando OpenAI Whisper (online ou offline)')
    parser.add_argument('input_source', help='Fonte de entrada (caminho do arquivo de vídeo ou URL do YouTube)')
    parser.add_argument('output_text', help='Caminho para o arquivo de texto de saída')
    parser.add_argument('--source-type', choices=['file', 'youtube'], default='file',
                       help='Tipo de fonte: arquivo ou URL do YouTube (padrão: file)')
    parser.add_argument('--start', type=float, help='Tempo inicial em segundos (opcional)')
    parser.add_argument('--end', type=float, help='Tempo final em segundos (opcional)')
    parser.add_argument('--with-timestamp', action='store_true', help='Incluir timestamps na transcrição')
    parser.add_argument('--segment-duration', type=int, default=600, help='Duração máxima de cada segmento de áudio em segundos (padrão: 600)')
    parser.add_argument('--mode', choices=['online', 'offline'], default='online',
                       help='Modo de transcrição: online (API) ou offline (local, requer transformers)')
    parser.add_argument('--silence-thresh', type=int, default=-40, help='Limiar de silêncio em dB (padrão: -40)')
    parser.add_argument('--min-silence-len', type=int, default=500, help='Duração mínima de silêncio em ms (padrão: 500)')
    parser.add_argument('--model', type=str, default="base",
                       help='Modelo Whisper para modo offline (padrão: base')
    args = parser.parse_args()
    
    input_source = args.input_source
    text_output_path = args.output_text
    source_type = args.source_type
    start_time = args.start
    end_time = args.end
    with_timestamp = args.with_timestamp
    segment_duration = args.segment_duration
    mode = args.mode
    silence_thresh = args.silence_thresh
    min_silence_len = args.min_silence_len
    model_name = args.model
    
    if (start_time is not None and start_time < 0) or (end_time is not None and end_time < 0):
        raise ValueError("Os tempos inicial e final devem ser não negativos")
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("O tempo inicial deve ser menor que o tempo final")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_path = temp_audio.name
    
    try:
        if source_type == 'youtube':
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = download_youtube_video(input_source, temp_dir)
                extract_audio_from_video(video_path, audio_path, start_time, end_time)
                asyncio.run(transcribe_audio_to_text(audio_path, text_output_path, video_url=input_source, with_timestamp=with_timestamp, segment_duration=segment_duration, mode=mode, silence_thresh=silence_thresh, min_silence_len=min_silence_len, model_name=model_name))
        else:  # file
            extract_audio_from_video(input_source, audio_path, start_time, end_time)
            asyncio.run(transcribe_audio_to_text(audio_path, text_output_path, with_timestamp=with_timestamp, segment_duration=segment_duration, mode=mode, silence_thresh=silence_thresh, min_silence_len=min_silence_len, model_name=model_name))
            
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

if __name__ == "__main__":
    main()