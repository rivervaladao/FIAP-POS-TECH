import cv2
import mediapipe as mp
import os
from tqdm import tqdm

def detect_pose_and_count_knee_movements(video_path, output_path):
    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo:", video_path)
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Contadores e flags para cada joelho
    left_knee_count = 0
    right_knee_count = 0
    left_knee_up = False
    right_knee_up = False

    def knee_is_up(knee_landmark, ankle_landmark):
        """
        Considera o joelho 'levantado' se a coordenada Y do joelho
        é menor que a do tornozelo. (No espaço de imagem, quanto menor o Y, mais acima está.)
        """
        return knee_landmark.y < ankle_landmark.y

    # Processar cada frame do vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Processar o frame para detectar a pose
        results = pose.process(rgb_frame)

        # Desenhar as anotações da pose no frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark

            # Verifica se o joelho esquerdo/direito está levantado
            current_left_knee_up = knee_is_up(
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            )
            current_right_knee_up = knee_is_up(
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            )

            # Se o joelho esquerdo acabou de passar de não-levantado para levantado, incrementa contador
            if current_left_knee_up and not left_knee_up:
                left_knee_count += 1

            # Se o joelho direito acabou de passar de não-levantado para levantado, incrementa contador
            if current_right_knee_up and not right_knee_up:
                right_knee_count += 1

            # Atualiza as flags
            left_knee_up = current_left_knee_up
            right_knee_up = current_right_knee_up

            # Exibir a contagem de elevações no frame
            cv2.putText(
                frame, 
                f'Left knee lifts: {left_knee_count}', 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            cv2.putText(
                frame, 
                f'Right knee lifts: {right_knee_count}', 
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

        # Exibir o frame processado
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Contagem final - Joelho Esquerdo: {left_knee_count}')
    print(f'Contagem final - Joelho Direito: {right_knee_count}')


if __name__ == '__main__':
    # Exemplo de caminhos de arquivo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, '../../data', 'moving.mp4')   # Altere para o nome do seu vídeo
    output_video_path = os.path.join(script_dir, '../../data', 'output_knee_up.mp4')  # Arquivo de saída

    detect_pose_and_count_knee_movements(input_video_path, output_video_path)
