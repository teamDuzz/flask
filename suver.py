from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import List
import json
import fasttext
import numpy as np


# 데이터 클래스 정의
@dataclass
class Profile:
    name: str
    schedule: List[List[int]]

@dataclass
class Mentor(Profile):
    major: str
    subjects: List[str]

@dataclass
class Mentee(Profile):
    interests: List[str]

@dataclass
class MentorshipData:
    mentors: List[Mentor]
    mentee: Mentee

# JSON -> 객체 변환 함수
def parse_mentorship_data(json_data: str) -> MentorshipData:
    data = json.loads(json_data)
    mentors = [
        Mentor(
            name=mentor['name'],
            schedule=mentor['schedule'],
            major=mentor['major'],
            subjects=mentor['subjects']
        )
        for mentor in data['mentors']
    ]
    mentee_data = data['mentee']
    mentee = Mentee(
        name=mentee_data['name'],
        schedule=mentee_data['schedule'],
        interests=mentee_data['interests']
    )
    return MentorshipData(mentors=mentors, mentee=mentee)
# FastText 모델 관련 함수
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

# 멘토-멘티 매칭 함수
def calculateFreeTimeScore(mentor: Mentor, mentee: Mentee) -> int:
    score = 0
    for day in range(7):
        for hour in range(0, 24):
            if mentor.schedule[day][hour] == 1 and mentee.schedule[day][hour] == 1:
                score += 1
    return score

# FastText 모델 경로 (미리 다운로드한 모델 경로를 설정하세요)
model_path = "/root/ai/flask/cc.ko.300.bin"  # 예: 영어 미리 학습된 모델
model = load_fasttext_model(model_path)

# Flask 서버 정의
app = Flask(__name__)

@app.route('/api/receive', methods=['POST'])
def receive_data():
    try:
        print("data received")
        json_data = request.data.decode('utf-8')
        mentorship_data = parse_mentorship_data(json_data)
        score=0
        best_match = None
        for mentor in mentorship_data.mentors:
            timescore = calculateFreeTimeScore(mentor, mentorship_data.mentee)
            interest_score=[]
            subject_score=[]
            for interest in mentorship_data.mentee.interests:
                tmp=get_word_similarity(mentor.major, interest, model)
                if(tmp>0.5):
                    tmp*=100
                elif(tmp>0.4):
                    tmp*=50
                elif(tmp>0.3):
                    tmp*=30
                else:
                    tmp*=10
                subject_score.append(tmp)
                for subject in mentor.subjects:
                    tmp=get_word_similarity(subject, interest, model)
                    if(tmp>0.5):
                        tmp*=100
                    elif(tmp>0.4):
                        tmp*=50
                    elif(tmp>0.3):
                        tmp*=30
                    else:
                        tmp*=10
                    interest_score.append(tmp)
            # 총 점수 계산(수정 해야함 )
            total_score = timescore + sum(interest_score) + sum(subject_score)
            if timescore == 0:
                total_score = 0
            if total_score > score:
                score = total_score
                best_match = mentor
            print(f"{mentor.name}: {total_score}")
        return jsonify({"status": "success", "data": best_match.__dict__}), 200   
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
