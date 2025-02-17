# duzz - 멘토-멘티 매칭 서버 (Flask)

## 개요
이 저장소는 **duzz** 프로젝트의 멘토-멘티 매칭을 위한 Flask 기반 서버입니다.  
사용자의 수강 데이터 및 관심사를 분석하고, **fastText**를 활용한 단어 유사도 기반 매칭 점수를 계산하여 최적의 멘토-멘티를 추천합니다.

## 주요 기능
- **fastText 기반 매칭 점수 계산**: 단어 유사도를 활용하여 멘토-멘티 적합도 분석
- **자동 추천 시스템**: 매칭 점수가 높은 사용자들을 자동으로 연결
- **RESTful API 제공**: 프론트엔드와의 통신을 위한 API 개발

## 기술 스택
- **Flask**: 백엔드 프레임워크
- **fastText**: 단어 임베딩 및 유사도 분석 ([공식 사이트](https://fasttext.cc/))
- **SQLAlchemy**: 데이터베이스 ORM
- **Pandas / NumPy**: 데이터 처리 및 분석
- **RESTful API**: 프론트엔드와의 데이터 통신

## 모델 다운로드
fastText 모델은 아래 링크에서 다운로드할 수 있습니다:  
[Pre-trained Word Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)


