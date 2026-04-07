# ML 프로젝트 환경 세팅 가이드

## 🎯 목표

-   로컬(집/회사)에서 코드 개발
-   GitHub로 코드 동기화
-   Colab에서 대규모 데이터 실행
-   Google Drive에 데이터/결과 저장

------------------------------------------------------------------------

## 🧱 전체 구조

VS Code (로컬) → GitHub (코드 관리) → Colab (실행) → Google Drive
(데이터 & 결과 저장)

------------------------------------------------------------------------

## 💻 로컬 환경 세팅 (집 / 회사 공통)

### 1. 필수 설치

-   Python (권장: 3.11)
-   Git
-   VS Code
-   Extensions: Python, Jupyter

### 2. GitHub 저장소 클론

    git clone /time-aware-behavior-prediction.git" href="https://github.com/<username>/time-aware-behavior-prediction.git" target="_blank">https://github.com/<username>/time-aware-behavior-prediction.git
    cd time-aware-behavior-prediction

### 3. 가상환경 생성

    py -3.11 -m venv .venv

### 4. 활성화

    .venv\Scripts\activate.bat

### 5. 패키지 설치

    python -m pip install --upgrade pip
    pip install -r requirements.txt

------------------------------------------------------------------------

## 🔁 작업 흐름

### 시작

    git pull

### 종료

    git add .
    git commit -m "update"
    git push

------------------------------------------------------------------------

## ☁️ Google Drive 구조

MyDrive/ai-projects/time-aware-behavior-prediction/

-   data/
-   outputs/
-   checkpoints/
-   logs/
-   notebooks/

------------------------------------------------------------------------

## 📓 Colab 설정

### Drive 연결

    from google.colab import drive
    drive.mount('/content/drive')

### 코드 실행

    !git clone <repo>
    !pip install -r requirements.txt
    !python src/train.py

------------------------------------------------------------------------

## ⚠️ 주의사항

-   .venv는 Git에 포함하지 않기
-   Python 버전 통일 (3.11 권장)
-   데이터는 Drive에 저장

------------------------------------------------------------------------

## 🎯 최종 구조

-   VS Code: 개발
-   GitHub: 코드 관리
-   Colab: 실행
-   Drive: 데이터/결과