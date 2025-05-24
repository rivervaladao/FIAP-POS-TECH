from moviepy import VideoFileClip
import os
import tempfile
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydub import AudioSegment
import yt_dlp

# Load environment variables from .env file
load_dotenv()

def download_youtube_video(url, output_path):
    """Download a YouTube video using yt-dlp and return the file path"""
    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'quiet': False,
            'merge_output_format': 'mp4',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading from: {url}")
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            print("YouTube download completed!")
            return downloaded_file
            
    except Exception as e:
        raise Exception(f"Error downloading YouTube video: {str(e)}")

def extract_audio_from_video(video_path, audio_path, start_time=None, end_time=None):
    video = VideoFileClip(video_path)
    if start_time is not None or end_time is not None:
        start_time = start_time if start_time is not None else 0
        end_time = end_time if end_time is not None else video.duration
        video = video.subclip(start_time, end_time)
        print(f"Processing video from {start_time}s to {end_time}s")
    with tqdm(total=100, desc="Extracting audio") as pbar:
        video.audio.write_audiofile(audio_path)
        pbar.update(100)

def split_audio(audio_path, max_size_mb=20):
    audio = AudioSegment.from_wav(audio_path)
    chunk_size_ms = 10 * 60 * 1000
    chunks = []
    max_size_bytes = max_size_mb * 1024 * 1024
    total_duration_ms = len(audio)
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        chunk_size_ms = int(total_duration_ms * (max_size_mb / file_size_mb))
    for i in range(0, len(audio), chunk_size_ms):
        chunk = audio[i:i + chunk_size_ms]
        chunk_path = f"{audio_path}_{i//1000}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio_to_text(audio_path, text_output_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    
    transcription_prompt = "Resuma em português do Brasil como manual técnico em Markdown"
    
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    max_size_mb = 20
    
    full_text = ""
    if file_size_mb > max_size_mb:
        print(f"Audio file ({file_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB). Splitting...")
        chunks = split_audio(audio_path)
    else:
        chunks = [audio_path]
    
    try:
        with tqdm(total=len(chunks), desc="Transcribing audio chunks") as pbar:
            for chunk_path in chunks:
                with open(chunk_path, 'rb') as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="pt",
                        prompt=transcription_prompt
                    )
                    chunk_text = transcription.text
                    if transcription_prompt in chunk_text:
                        chunk_text = chunk_text.replace(transcription_prompt, "").strip()
                    full_text += chunk_text + "\n\n"
                    pbar.update(1)
        
        print("Transcrição:\n" + full_text)
        with open(text_output_path, 'w', encoding='utf-8') as file:
            file.write(full_text.strip())
    
    except Exception as e:
        print(f"Erro ao transcrever o áudio com Whisper: {str(e)}")
    finally:
        for chunk in chunks[1:]:
            if os.path.exists(chunk):
                os.unlink(chunk)

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio from video using OpenAI Whisper')
    parser.add_argument('input_source', help='Input source (path to video file or YouTube URL)')
    parser.add_argument('output_text', help='Path for the output text transcription')
    parser.add_argument('--source-type', choices=['file', 'youtube'], default='file',
                       help='Source type: file system or YouTube URL (default: file)')
    parser.add_argument('--start', type=float, help='Start time in seconds (optional)')
    parser.add_argument('--end', type=float, help='End time in seconds (optional)')
    args = parser.parse_args()
    
    input_source = args.input_source
    text_output_path = args.output_text
    source_type = args.source_type
    start_time = args.start
    end_time = args.end
    
    if (start_time is not None and start_time < 0) or (end_time is not None and end_time < 0):
        raise ValueError("Start and end times must be non-negative")
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("Start time must be less than end time")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_path = temp_audio.name
    
    try:
        if source_type == 'youtube':
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = download_youtube_video(input_source, temp_dir)
                extract_audio_from_video(video_path, audio_path, start_time, end_time)
                transcribe_audio_to_text(audio_path, text_output_path)
        else:  # file
            extract_audio_from_video(input_source, audio_path, start_time, end_time)
            transcribe_audio_to_text(audio_path, text_output_path)
            
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

if __name__ == "__main__":
    main()