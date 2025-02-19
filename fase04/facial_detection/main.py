from deepface import DeepFace
result = DeepFace.analyze(img_path="./data/emotion01.jpeg", actions=['emotion'])
print(result)
result = DeepFace.analyze(img_path="./data/emotion02.jpeg", actions=['emotion'])
print(result)