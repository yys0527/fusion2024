---
tags: [project, nlp, embedding, filter-bubble]
aliases: [fusion2024, 알고리즘 편향도 평가 서비스]
created: 2024-07
---

> [!NOTE]
> **AI 사용 내역**: 프로젝트 코드·데이터·보고서는 직접 작성. AI(Claude)는 `README.md` 작성에만 사용

# fusion2024

임베딩 모델을 통한 알고리즘 편향도 평가 서비스

> [!NOTE]
> 추천 알고리즘이 어떤 주제의 콘텐츠를 얼마나 치우치게 노출하는지 수치로 측정하는 프로젝트
> 웹 서비스로 배포 (AWS EC2 + Django, 제목 입력 → 관련 분야 출력)

## 개발 동기

- SNS·쇼핑몰·유튜브 등 개인화 추천 알고리즘 보편화
- 관심 주제 위주의 반복 노출 → 좁은 정보 범위, **필터 버블** 발생
- 확증 편향 강화 → 건전한 사고 저해
- 추천 콘텐츠를 분석해 편향도를 측정하는 서비스로 해결 시도

## 사전 조사

- 교내 학생 18명 설문
  - 필터 버블 인지 3명
  - 필터 버블 경험 18명 (전원)
  - 경험 플랫폼: 유튜브 > SNS > 쇼핑몰
- 경험 빈도가 가장 높은 **유튜브**를 대상 플랫폼으로 선정

## 작동 원리

1. 콘텐츠 제목 수집 (배포 버전: 사용자 직접 입력)
2. 전처리 (괄호·특수문자·이모지 제거)
3. 비지도 토크나이저(soynlp)로 토큰화
4. OpenAI 임베딩 (`text-embedding-3-large`, 3072차원)
5. 미리 임베딩해 둔 주제(정치, 동물 등)와 **코사인 유사도** 계산
6. 토큰별 유사도 평균 → 제목의 주제 판별
7. 다수 콘텐츠에 반복 → 주제별 추천 비중 산출
8. 특정 주제가 일정 수준 이상이면 **편향**으로 판단

```
제목 → 전처리 → 토큰화 → 임베딩 → 카테고리 유사도 → 주제별 비중 → 편향도
```

## 배포

> [!NOTE]
> 구글 로그인으로 영상 목록을 자동 수집하는 기능은 구글 측 프로젝트 승인 불가로 미구현 → 제목 입력 시 관련 분야를 출력하는 방식으로 구현

### 구성

| 역할 | 서비스 |
| --- | --- |
| 메인 서버 (페이지 서빙) | AWS EC2 + Django |
| API 서버 (임베딩 처리) | 개인 서버 + Flask |
| API 중계·호출 로깅 | AWS Lambda |
| 토큰 저장 | AWS RDS |
| 카테고리 임베딩 데이터 | Supabase |
| 임베딩 모델 | OpenAI |

- 보안 강화: EC2에서 API 서버를 직접 호출하지 않고 **Lambda 경유**
- 접근 토큰: 요청마다 RDS에 저장 → 응답 수신 후 삭제

### 호출 흐름

- **1** 클라이언트가 제목 입력
- **2·3** EC2가 요청 수신 → 클라이언트 검증(정상 경로 호출 여부, 사람 여부) → RDS에 토큰 저장
- **4** 제목·토큰을 파라미터로 Lambda 호출
- **5** Lambda가 호출 기록 후 개인 서버 API 호출
- **6** Supabase 임베딩 데이터 + OpenAI 임베딩 모델 호출
- **7** 응답 데이터를 개인 서버에서 처리
- **8** 토큰·API 호출 정보를 개인 서버에 저장, EC2에 토큰·응답 데이터 전달
- **9·10** 응답 정상 수신 시 RDS에서 토큰 삭제
- **11** 클라이언트에 결과 출력

```
Client ⇄ EC2 (Django) ⇄ RDS (토큰)
            │
            ▼
         Lambda (로깅) ──▶ 개인 서버 (Flask) ⇄ Supabase / OpenAI
```

- 상세 다이어그램: [배포 보고서](Document/배포보고서.pdf)

## 구조

```
fusion2024/
├── Code/
│   ├── tokenizer.py     # 전처리 + soynlp MaxScoreTokenizer
│   └── similarity.py    # 임베딩, 카테고리 유사도 계산
├── Data/
│   └── corpus.txt       # 토크나이저 학습용 말뭉치 (영화 리뷰)
└── Document/
    ├── 보고서.pdf
    ├── 발표자료.pdf
    └── 배포보고서.pdf
```

| 파일 | 역할 |
| --- | --- |
| `Code/tokenizer.py` | 문장 전처리, 말뭉치 기반 단어 추출·학습(`WordExtractor`), 응집도(cohesion) 점수 기반 토큰화 |
| `Code/similarity.py` | 토큰 임베딩, DB의 카테고리 임베딩과 코사인 유사도 계산, 유사도순 정렬 출력 |
| `Data/corpus.txt` | 약 79,000행 한국어 영화 리뷰 (단어 추출 학습용) |

## 실행 환경

- Python 3
- 의존성

```bash
pip install numpy soynlp openai supabase
```

- 외부 서비스
  - OpenAI API (임베딩)
  - Supabase (`category` 테이블: `title`, `embedding`)

## 설정

> [!WARNING]
> `Code/private_data.py`는 `.gitignore` 대상. 직접 생성 필요 (API 키 커밋 금지)

```python
# Code/private_data.py
bpath = "<프로젝트 루트 경로>/"   # corpus.txt 경로 계산용
aikey = "<OPENAI_API_KEY>"
dburl = "<SUPABASE_URL>"
dbkey = "<SUPABASE_KEY>"
```

- `tokenizer.py`는 `bpath + "data/corpus.txt"`를 읽음 → 폴더명 대소문자(`Data/`) 확인

## 향후 계획

- 구글 로그인 연동: 영상 목록 자동 수집 (구글 승인 필요)
- 제목 외 정보 반영: 설명, 태그 → 텍스트 정보 통합 판단
- 이미지 모델로 썸네일 분석 추가
- 정확도 개선

## 문서

- [보고서](Document/보고서.pdf)
- [발표자료](Document/발표자료.pdf)
- [배포 보고서](Document/배포보고서.pdf)

## 제작

- 이수민
- 윤영서 ([@yys0527](https://github.com/yys0527))
