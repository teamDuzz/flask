from flask import Flask, Response, request, jsonify
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

    def __hash__(self):
        return hash((self.name, self.major, tuple(self.subjects)))

    def __eq__(self, other):
        if not isinstance(other, Mentor):
            return False
        return (self.name, self.major, self.subjects) == (other.name, other.major, other.subjects)


@dataclass
class Mentee(Profile):
    interests: List[str]
    subjects: List[str]
    option : bool


@dataclass
class MentorshipData:
    mentors: List[Mentor]
    mentees: List[Mentee]

def custom_serializer(obj):
    if isinstance(obj, Mentor) or isinstance(obj, Mentee):
        return obj.__dict__
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


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
    mentees = [
        Mentee(
            name=mentee['name'],
            schedule=mentee['schedule'],
            interests=mentee['interests'],
            subjects=mentee['subjects'],
            option=mentee['option']
        )
        for mentee in data['mentees']
    ]
    return MentorshipData(mentors=mentors, mentees=mentees)


def cosine_similarity(vector1, vector2):
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    if norm_vector1 == 0 or norm_vector2 == 0:  # 벡터 길이가 0인 경우
        return 0  # 유사도를 0으로 반환
    return dot_product / (norm_vector1 * norm_vector2)


def load_fasttext_model(model_path):
    """FastText 모델을 로드합니다."""
    return fasttext.load_model(model_path)


def get_word_similarity(word1, word2, model):
    """두 단어의 유사도를 계산합니다."""
    vector1 = model.get_word_vector(word1)
    vector2 = model.get_word_vector(word2)
    return cosine_similarity(vector1, vector2)

def adjust_score(score) ->int:
    if score < 0:
        return 0
    elif score <0.3:
        return score*0.3
    elif score <0.4:
        return score*0.6
    elif score <0.6:
        return  score*0.8
    elif score <0.8:
        return score*0.9
    else:
        return score
    return score
# 멘토-멘티 매칭 함수
def calculateFreeTimeScore(mentor: Mentor, mentee: Mentee) -> int:
    score = 0
    for day in range(7):
        for hour in range(0, 48):
            if mentor.schedule[day][hour] == 1 and mentee.schedule[day][hour] == 1:
                score += 1
    return score

def calculateInterestScore(mentor: Mentor, mentee: Mentee) -> int:
    score = caltime(mentor, mentee) + 2*calinterest(mentor, mentee) #최대 0~5점
    return score

def calculateTimeScore(mentor: Mentor, mentee: Mentee) -> int:
    score= 2*caltime(mentor, mentee) + calinterest(mentor, mentee) #최대 0~5점
    return score

def caltime(mentor: Mentor, mentee: Mentee) -> int:
    time_score=0
    for day in range(0,5):
        for hour in range(18, 42):
            if mentor.schedule[day][hour] == 1 and mentee.schedule[day][hour] == 1:
                time_score += 0.5
    if time_score>=15:
        time_score=15
    time_score/=15 #최대 0~1점
    subject_score = 0
    for subject in mentee.subjects:
        similarity_score = get_word_similarity(subject,mentor.major, model)
        subject_score += adjust_score(similarity_score)
    subject_score/=len(mentee.subjects) #최대 0~1점
    score = time_score + subject_score #최대 0~2점
    return score

def calinterest(mentor: Mentor, mentee: Mentee) -> int:
    subject_score=0
    major_score=0
    for interest in mentee.interests:
        similarity_score = get_word_similarity(interest,mentor.major, model)
        major_score += adjust_score(similarity_score)
        for subject in mentor.subjects:
            similarity_score = get_word_similarity(subject,interest, model)
            similarity_score += adjust_score(similarity_score)
    major_score/=len(mentee.interests) #최대 0~1점
    subject_score/=len(mentee.interests)*len(mentor.subjects) #최대 0~1점
    score = major_score + subject_score #최대 0~2점
    return score



# FastText 모델 경로 (미리 다운로드한 모델 경로를 설정하세요)
model_path = "C:\\Users\\gjw19\\PycharmProjects\\flask\\cc.ko.300.bin"  # 예: 영어 미리 학습된 모델
model = load_fasttext_model(model_path)

# Flask 서버 정의
app = Flask(__name__)


@app.route('/api/receive', methods=['POST'])
def receive_data():
    try:
        print("Data received")
        json_data = request.data.decode('utf-8')
        mentorship_data = parse_mentorship_data(json_data)
        mentornum = len(mentorship_data.mentors)
        menteenum = len(mentorship_data.mentees)
        score_array = [[0] * menteenum for _ in range(mentornum)]
        
        # 점수 계산
        for i, mentor in enumerate(mentorship_data.mentors):
            for j, mentee in enumerate(mentorship_data.mentees):
                if mentee.option:
                    score_array[i][j] = calculateInterestScore(mentor, mentee)
                else:
                    score_array[i][j] = calculateFreeTimeScore(mentor, mentee)

        # 멘토당 최대 n명의 멘티 배정
        n = 5
        assigned_mentees = {mentor: [] for mentor in mentorship_data.mentors}
        mentee_assigned = [False] * menteenum

        for _ in range(menteenum):
            max_score = -1
            best_mentor_idx = -1
            best_mentee_idx = -1

            for i, mentor in enumerate(mentorship_data.mentors):
                if len(assigned_mentees[mentor]) < n:
                    for j, mentee in enumerate(mentorship_data.mentees):
                        if not mentee_assigned[j] and score_array[i][j] > max_score:
                            max_score = score_array[i][j]
                            best_mentor_idx = i
                            best_mentee_idx = j

            if best_mentor_idx != -1 and best_mentee_idx != -1:
                assigned_mentees[mentorship_data.mentors[best_mentor_idx]].append(
                    mentorship_data.mentees[best_mentee_idx]
                )
                mentee_assigned[best_mentee_idx] = True
        for mentor, mentees in assigned_mentees.items():
            print(f"{mentor.name} -> {[mentee.name for mentee in mentees]}")
        # JSON 응답 생성
        response = {
            "status": "success",
            "result": [
                {
                    "mentor": {
                        "name": mentor.name,
                        "schedule": mentor.schedule,
                        "major": mentor.major,
                        "subjects": mentor.subjects,
                    },
                    "mentees": [
                        {
                            "name": mentee.name,
                            "schedule": mentee.schedule,
                            "interests": mentee.interests,
                            "subjects": mentee.subjects,
                            "option": mentee.option,
                        }
                        for mentee in mentees
                    ] if mentees else []  # 멘티가 없으면 빈 리스트 반환
                }
                for mentor, mentees in assigned_mentees.items()
            ],
        }
        print("Data processed")
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
