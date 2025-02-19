#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aplicação para análise de vídeo:
- Detecção facial com MediaPipe
- Análise de emoções com DeepFace
- Detecção de atividades com base em landmarks de pose (heurísticas)
- Geração de relatório com estatísticas, timeline de eventos e vídeo anotado
- Clusterização dos embeddings faciais com sklearn (DBSCAN)
- Extração de embeddings faciais face_recognition

Como usar:
    python video_analyzer.py caminho/para/o/video.mp4

Requisitos:
    - sklearn
    - opencv-python
    - mediapipe
    - deepface
    - face_recognition
    - tqdm
    - numpy
    - string
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import string
from deepface import DeepFace
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
import face_recognition

# Classe para detecção de pose usando MediaPipe
class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, frame):
        # Converte o frame para RGB e processa com MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

# Classe para detecção de mãos usando MediaPipe
class HandsDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, frame):
        # Converte o frame para RGB e processa com MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

# Classe para detecção de face com Mesh usando MediaPipe
class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=7,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        # Converte o frame para RGB e processa com MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)

# Classe que contém a lógica de classificação de situações (pose, mãos, emoções e aperto de mãos)
class SituationClassifier:
    def __init__(self):
        self.last_pose_class = "Parado"
        self.last_hands_class = "Nenhuma mão"
        self.last_face_class = "Neutro"
        self.pose_miss_count = 0
        self.pose_miss_limit = 10

        # Armazena histórico de posições para analisar movimentos suaves (ex: dança)
        self.last_shoulder_positions = []
        self.last_hand_positions = []

    def classify_pose(self, pose_landmarks):
        # Caso não haja landmarks, incrementa contador e retorna "Sem Pose" se ultrapassar limite
        if not pose_landmarks:
            self.pose_miss_count += 1
            if self.pose_miss_count >= self.pose_miss_limit:
                return "Sem Pose"
            return self.last_pose_class

        self.pose_miss_count = 0

        # Utiliza landmarks específicos para verificar se as pernas estão em movimento
        nariz = pose_landmarks[0]
        pe_esquerdo = pose_landmarks[27]
        pe_direito = pose_landmarks[28]

        pernas_em_movimento = abs(pe_esquerdo.y - pe_direito.y) > 0.1
        situacao = "Andando" if pernas_em_movimento else "Parado"

        self.last_pose_class = situacao
        return situacao

    def classify_hands(self, hands_results):
        # Se nenhuma mão for detectada, retorna "Nenhuma mão"
        if not hands_results.multi_hand_landmarks:
            return "Nenhuma mão"
        return "Movimento das mãos"

    def classify_dance(self, pose_landmarks):
        """
        Detecta dança com base em movimentos suaves e coordenados do tronco e das mãos.
        """
        if not pose_landmarks:
            return False

        # Seleciona alguns landmarks importantes para a análise
        nariz = pose_landmarks[0]
        ombro_esquerdo = pose_landmarks[11]
        ombro_direito = pose_landmarks[12]
        cotovelo_esquerdo = pose_landmarks[13]
        cotovelo_direito = pose_landmarks[14]
        punho_esquerdo = pose_landmarks[15]
        punho_direito = pose_landmarks[16]

        # Cria arrays com posições do tronco e mãos
        tronco_posicao = np.array([(nariz.x, nariz.y), (ombro_esquerdo.x, ombro_esquerdo.y), (ombro_direito.x, ombro_direito.y)])
        maos_posicao = np.array([(cotovelo_esquerdo.x, cotovelo_esquerdo.y), (cotovelo_direito.x, cotovelo_direito.y),
                                 (punho_esquerdo.x, punho_esquerdo.y), (punho_direito.x, punho_direito.y)])

        # Mantém um histórico com tamanho máximo de 10 frames
        if len(self.last_shoulder_positions) > 10:
            self.last_shoulder_positions.pop(0)
            self.last_hand_positions.pop(0)

        self.last_shoulder_positions.append(tronco_posicao)
        self.last_hand_positions.append(maos_posicao)

        # Se o histórico ainda não é suficiente, não classifica como dança
        if len(self.last_shoulder_positions) < 5:
            return False

        # Calcula variação padrão para avaliar o movimento suave
        tronco_variacao = np.std([np.mean(pos, axis=0) for pos in self.last_shoulder_positions], axis=0)
        maos_variacao = np.std([np.mean(pos, axis=0) for pos in self.last_hand_positions], axis=0)

        movimento_suave_tronco = tronco_variacao[0] > 0.015 or tronco_variacao[1] > 0.015
        movimento_suave_maos = maos_variacao[0] > 0.05 or maos_variacao[1] > 0.05

        return movimento_suave_tronco and movimento_suave_maos

    def classify_emotion(self, face_landmarks_list, frame):
        """
        Utiliza o DeepFace para classificar a emoção predominante em um ou mais rostos.
        """
        if not face_landmarks_list:
            return "Neutro"

        h, w, _ = frame.shape
        emotions_detectadas = []

        for face_landmarks in face_landmarks_list:
            # Define as coordenadas para recortar o rosto a partir dos landmarks
            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]

            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w, max(xs)), min(h, max(ys))

            face_crop = frame[y1:y2, x1:x2]

            # Ignora recortes muito pequenos
            if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                continue

            try:
                # Analisa a emoção usando DeepFace
                result = DeepFace.analyze(
                    img_path=face_crop,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='mediapipe'
                )

                if isinstance(result, list):
                    result = result[0]

                # Seleciona a emoção com maior probabilidade
                emotion = max(result['emotion'], key=result['emotion'].get)
                emotions_detectadas.append(emotion)

            except Exception as e:
                print(f"[ERRO DeepFace]: {e}")

        if emotions_detectadas:
            return Counter(emotions_detectadas).most_common(1)[0][0]
        else:
            return "Neutro"

    def classify_handshake(self, hands_results):
        """
        Detecta aperto de mãos considerando a proximidade entre duas mãos.
        """
        if not hands_results.multi_hand_landmarks or len(hands_results.multi_hand_landmarks) < 2:
            return False

        palmas = []
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Utiliza o landmark 0 que representa a palma da mão
            palma_x = hand_landmarks.landmark[0].x
            palma_y = hand_landmarks.landmark[0].y
            palmas.append((palma_x, palma_y))

        if len(palmas) < 2:
            return False

        distancias = []
        for i in range(len(palmas)):
            for j in range(i + 1, len(palmas)):
                dist = np.linalg.norm(np.array(palmas[i]) - np.array(palmas[j]))
                distancias.append(dist)

        LIMIAR_PROXIMIDADE = 0.05

        # Verifica se alguma distância está abaixo do limiar, indicando aperto de mãos
        for dist in distancias:
            if dist < LIMIAR_PROXIMIDADE:
                return True

        return False

# Classe principal para análise do vídeo
class VideoAnalyzer:
    def __init__(self, video_path, frame_skip=1):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.pose_detector = PoseDetector()
        self.hands_detector = HandsDetector()
        self.face_mesh_detector = FaceMeshDetector()
        self.classifier = SituationClassifier()

        self.frames_analisados = 0
        self.anomalias_detectadas = 0
        self.tipos_anomalias = Counter()
        # Ocorrências registradas para cada evento único (atividade)
        self.ocorrencias = defaultdict(int)
        self.emocoes_contagem = defaultdict(int)

        self.pessoas_detectadas_por_frame = []
        self.embeddings = []

        self.WINDOW_SIZE = 10  # Tamanho da janela para agregação dos frames
        self.frame_buffer = []  # Armazena as classificações de uma janela de frames
        self.atividades_timeline = defaultdict(list)  # Armazena os intervalos de tempo dos eventos
        self.ultima_atividade = None
        self.atividade_inicio_tempo = None
        self.annotated_frames_info = []  # Armazena as informações de anotações dos rostos para o vídeo anotado

    def analyze(self):
        """
        Processa o vídeo frame a frame, aplicando detecções e classificações.
        """
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc="Processando vídeo", unit="frame") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"[ERRO] Falha ao ler frame na posição {frame_count}")
                    break

                frame_count += 1
                # Processa somente frames de acordo com o frame_skip
                if frame_count % self.frame_skip != 0:
                    pbar.update(1)
                    continue

                self.frames_analisados += 1

                # Processa detecções de pose, mãos e faces
                pose_situacao = self.process_pose(frame)
                hands_situacao = self.process_hands(frame)
                _, num_faces = self.process_faces(frame)

                self.pessoas_detectadas_por_frame.append(num_faces)
                self.frame_buffer.append((pose_situacao, hands_situacao))

                # Quando o buffer atinge o tamanho definido, agrega os resultados
                if len(self.frame_buffer) == self.WINDOW_SIZE:
                    self.aggregate_and_register_buffer()

                pbar.update(1)

            # Finaliza a timeline registrando a última atividade
            tempo_final_video_segundos = (self.frames_analisados * self.frame_skip) / 30.0
            if self.ultima_atividade is not None and self.atividade_inicio_tempo is not None:
                self.atividades_timeline[self.ultima_atividade].append({
                    'inicio': self.atividade_inicio_tempo,
                    'fim': tempo_final_video_segundos
                })

            if len(self.frame_buffer) > 0:
                self.aggregate_and_register_buffer()

        cap.release()

    def process_pose(self, frame):
        # Detecta pose e retorna a classificação
        pose_results = self.pose_detector.detect(frame)
        if pose_results and pose_results.pose_landmarks:
            pose_situacao = self.classifier.classify_pose(pose_results.pose_landmarks.landmark)
            # Verifica se há movimento de dança
            if self.classifier.classify_dance(pose_results.pose_landmarks.landmark):
                pose_situacao = "Dançando"
            return pose_situacao
        return self.classifier.classify_pose(None)
    
    def process_hands(self, frame):
        # Processa detecção de mãos e verifica se há aperto de mãos
        hands_results = self.hands_detector.detect(frame)
        situacao = self.classifier.classify_hands(hands_results)
        if self.classifier.classify_handshake(hands_results):
            situacao = "Aperto de mãos"
        return situacao

    def process_faces(self, frame):
        # Processa detecção de faces e retorna contagem de faces
        face_results = self.face_mesh_detector.detect(frame)
        if not face_results or not face_results.multi_face_landmarks:
            return "Neutro", 0

        num_faces = len(face_results.multi_face_landmarks)
        faces_info = []
        for face_landmarks in face_results.multi_face_landmarks:
            face_crop = self.extract_face_crop(frame, face_landmarks)
            # Verifica se o recorte do rosto é válido
            if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                continue
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(face_rgb)
            if encodings:
                self.embeddings.append(encodings[0])
            # Classifica a emoção do rosto
            emotion = self.classifier.classify_emotion([face_landmarks], frame)
            emotions_pt = {
                "happy": "Feliz",
                "sad": "Triste",
                "fear": "Medo",
                "surprise": "Surpreso",
                "neutral": "Neutro",
                "angry": "Raiva",
                "disgust": "Nojo",
            }
            emotion_traduzida = emotions_pt.get(emotion, "Neutro")
            self.emocoes_contagem[emotion_traduzida] += 1
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w, max(xs)), min(h, max(ys))
            faces_info.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'emotion': emotion_traduzida
            })
        self.annotated_frames_info.append(faces_info)
        return "Neutro", num_faces

    def draw_face_annotations(self, frame, faces_info):
        # Desenha caixas e textos para cada rosto detectado
        for face in faces_info:
            x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
            emotion = face['emotion']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def save_annotated_video(self, output_path='video_anotado.mp4'):
        """
        Salva o vídeo com as anotações (caixas e emoções) sobre os rostos.
        """
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count < len(self.annotated_frames_info):
                faces_info = self.annotated_frames_info[frame_count]
                self.draw_face_annotations(frame, faces_info)
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
        print(f"Vídeo anotado salvo em {output_path}")

    def aggregate_and_register_buffer(self):
        """
        Agrega as classificações do buffer e registra um evento único somente
        quando a atividade detectada mudar.
        """
        final_pose, final_hands = self.aggregate_classifications(self.frame_buffer)
        frame_base = self.frames_analisados - self.WINDOW_SIZE
        tempo_atual_segundos = (frame_base * self.frame_skip) / 30.0

        # Registra um novo evento se a atividade atual for diferente da última
        if self.ultima_atividade != final_pose:
            if self.ultima_atividade is not None and self.atividade_inicio_tempo is not None:
                self.atividades_timeline[self.ultima_atividade].append({
                    'inicio': self.atividade_inicio_tempo,
                    'fim': tempo_atual_segundos
                })
            self.ultima_atividade = final_pose
            self.atividade_inicio_tempo = tempo_atual_segundos
            self.ocorrencias[final_pose] += 1  # Evento único registrado
            if final_pose == "Sem Pose":
                self.anomalias_detectadas += 1
                self.tipos_anomalias["Anomalia"] += 1

        # Limpa o buffer após a agregação
        self.frame_buffer = []

    def aggregate_classifications(self, buffer_data):
        # Agrega as classificações de pose e mãos usando votação majoritária
        pose_list = [x[0] for x in buffer_data]
        hands_list = [x[1] for x in buffer_data]
        final_pose = Counter(pose_list).most_common(1)[0][0]
        final_hands = Counter(hands_list).most_common(1)[0][0]
        return final_pose, final_hands

    def cluster_faces(self):
        """
        Utiliza DBSCAN para agrupar embeddings dos rostos e determinar o número de pessoas únicas.
        """
        if not self.embeddings:
            print("Nenhum rosto foi detectado!")
            return 0
        embeddings_array = np.array(self.embeddings)
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='euclidean').fit(embeddings_array)
        labels = clustering.labels_
        unique_faces = len(set(labels) - {-1})
        return unique_faces

    def extract_face_crop(self, frame, face_landmarks):
        """
        Extrai o recorte do rosto a partir dos landmarks.
        """
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]

    def generate_summary(self):
        """
        Gera um relatório com o resumo da análise:
          - Total de frames analisados
          - Máximo de pessoas detectadas em um frame
          - Número de pessoas únicas
          - Número de anomalias detectadas
          - Eventos únicos de atividades
          - Contagem de emoções
          - Timeline das atividades
        """
        max_pessoas = max(self.pessoas_detectadas_por_frame) if self.pessoas_detectadas_por_frame else 0
        pessoas_unicas = self.cluster_faces()
        resumo_atividades = "\n".join([f"{atividade}: {qtd} eventos" for atividade, qtd in self.ocorrencias.items()])
        resumo_emocoes = "\n".join([f"{emocao}: {qtd} vezes" for emocao, qtd in self.emocoes_contagem.items()])

        tempo_minimo_danca = 5.0  # Filtro para ignorar eventos curtos de dança
        resumo_timeline = []
        for atividade, intervalos in self.atividades_timeline.items():
            for intervalo in intervalos:
                duracao = intervalo['fim'] - intervalo['inicio']
                if atividade == "Dançando" and duracao < tempo_minimo_danca:
                    continue
                inicio = f"{int(intervalo['inicio'] // 60)}m{int(intervalo['inicio'] % 60)}s"
                fim = f"{int(intervalo['fim'] // 60)}m{int(intervalo['fim'] % 60)}s"
                resumo_timeline.append(f"{atividade} de {inicio} até {fim}")
        resumo_timeline_str = "\n".join(resumo_timeline)

        with open('resumo_video.txt', 'w') as file:
            file.write(f"Total de frames analisados: {self.frames_analisados}\n")
            file.write(f"Máximo de pessoas detectadas em um frame: {max_pessoas}\n")
            file.write(f"Pessoas únicas identificadas: {pessoas_unicas}\n")
            file.write(f"Anomalias detectadas: {self.anomalias_detectadas}\n")
            file.write("\nAtividades detectadas:\n")
            file.write(f"{resumo_atividades}\n")
            file.write("\nEmoções detectadas:\n")
            file.write(f"{resumo_emocoes}\n")
            file.write("\nTimeline das atividades:\n")
            file.write(f"{resumo_timeline_str}\n")

def main():
    parser = argparse.ArgumentParser(description="Aplicação de análise de vídeo")
    parser.add_argument("video_path", help="Caminho para o arquivo de vídeo")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Arquivo {args.video_path} não encontrado.")
        return

    try:
        analyzer = VideoAnalyzer(args.video_path, frame_skip=1)
    except ValueError as e:
        print(e)
        return

    analyzer.analyze()
    analyzer.generate_summary()
    analyzer.save_annotated_video('video_anotado.mp4')

if __name__ == "__main__":
    main()