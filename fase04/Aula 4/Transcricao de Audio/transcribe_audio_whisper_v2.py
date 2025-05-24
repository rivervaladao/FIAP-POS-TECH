from moviepy import VideoFileClip
import os
import tempfile
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydub import AudioSegment
import yt_dlp
import urllib.parse
import asyncio

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configura o cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def split_audio(audio_path, segment_duration=600):  # 10 minutos por padrão
    """Divide o áudio em segmentos com duração especificada"""
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    for i in range(0, len(audio), segment_duration * 1000):
        start_ms = i
        end_ms = min(i + segment_duration * 1000, len(audio))
        chunk = audio[start_ms:end_ms]
        chunk_path = f"{audio_path}_segment_{start_ms//1000}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_ms / 1000, end_ms / 1000))
    return chunks

async def transcribe_segment(segment_path):
    """Transcreve um segmento de áudio usando a API Whisper"""
    try:
        # Usamos o caminho do arquivo diretamente, sem abrir com aiofiles
        with open(segment_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="pt"
            )
        return transcription.text
    except Exception as e:
        print(f"Erro ao transcrever segmento {segment_path}: {str(e)}")
        return None

async def transcribe_audio_to_text(audio_path, text_output_path, video_url=None, with_timestamp=True, segment_duration=600):
    """Transcreve o áudio para texto usando a API Whisper com processamento assíncrono"""
    chunks = split_audio(audio_path, segment_duration=segment_duration)
    tasks = []
    for chunk_path, start_time, end_time in chunks:
        tasks.append(transcribe_segment(chunk_path))
    
    results = await asyncio.gather(*tasks)
    
    full_text = ""
    for idx, (chunk_path, start_time, end_time) in enumerate(chunks):
        chunk_text = results[idx]
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
    
    # Salva a transcrição completa
    with open(text_output_path, 'w', encoding='utf-8') as file:
        file.write(full_text.strip())
    
    # Limpa os arquivos temporários
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
    parser = argparse.ArgumentParser(description='Transcreve áudio de vídeo usando OpenAI Whisper')
    parser.add_argument('input_source', help='Fonte de entrada (caminho do arquivo de vídeo ou URL do YouTube)')
    parser.add_argument('output_text', help='Caminho para o arquivo de texto de saída')
    parser.add_argument('--source-type', choices=['file', 'youtube'], default='file',
                       help='Tipo de fonte: arquivo ou URL do YouTube (padrão: file)')
    parser.add_argument('--start', type=float, help='Tempo inicial em segundos (opcional)')
    parser.add_argument('--end', type=float, help='Tempo final em segundos (opcional)')
    parser.add_argument('--with-timestamp', action='store_true', help='Incluir timestamps na transcrição')
    parser.add_argument('--segment-duration', type=int, default=600, help='Duração de cada segmento de áudio em segundos (padrão: 600)')
    args = parser.parse_args()
    
    input_source = args.input_source
    text_output_path = args.output_text
    source_type = args.source_type
    start_time = args.start
    end_time = args.end
    with_timestamp = args.with_timestamp
    segment_duration = args.segment_duration
    
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
                asyncio.run(transcribe_audio_to_text(audio_path, text_output_path, video_url=input_source, with_timestamp=with_timestamp, segment_duration=segment_duration))
        else:  # file
            extract_audio_from_video(input_source, audio_path, start_time, end_time)
            asyncio.run(transcribe_audio_to_text(audio_path, text_output_path, with_timestamp=with_timestamp, segment_duration=segment_duration))
            
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

if __name__ == "__main__":
    main()