import fasttext
import numpy as np
def cosine_similarity(vector1, vector2):
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)

def load_fasttext_model(model_path):
    """FastText 모델을 로드합니다."""
    return fasttext.load_model(model_path)

def get_word_similarity(word1, word2, model):
    """두 단어의 유사도를 계산합니다."""
    vector1 = model.get_word_vector(word1)
    vector2 = model.get_word_vector(word2)
    return cosine_similarity(vector1, vector2)

# FastText 모델 경로 (미리 다운로드한 모델 경로를 설정하세요)
model_path = "C:\\Users\\gjw19\\PycharmProjects\\flask\\cc.ko.300.bin"  # 예: 영어 미리 학습된 모델
model = load_fasttext_model(model_path)

# 두 단어를 입력받고 유사도 계산d
while True:
    word1 = input("첫 번째 단어를 입력하세요 (종료하려면 'exit' 입력): ")
    if word1.lower() == 'exit':
        break
    word2 = input("두 번째 단어를 입력하세요 (종료하려면 'exit' 입력): ")
    if word2.lower() == 'exit':
        break
    similarity_score = get_word_similarity(word1, word2, model)
    print(f"'{word1}'와(과) '{word2}'의 유사도 점수: {similarity_score:.4f}")
