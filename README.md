# Wine Class Classification Predictor

## Website

배포 링크:  
https://wineclassification-production.up.railway.app/gradio

## 프로젝트 소개

이 프로젝트는 `scikit-learn`의 `load_wine()` 데이터를 사용하여 와인의 클래스를 분류하는 머신러닝 모델입니다.

사용자는 3개의 화학적 특성값을 입력하면 모델이 와인의 클래스를 예측합니다.

## 사용한 기술

- Python
- scikit-learn
- FastAPI
- Gradio
- Uvicorn
- NumPy

## 데이터셋

이 프로젝트는 `scikit-learn`에 내장된 wine dataset을 사용했습니다.

```python
from sklearn.datasets import load_wine
