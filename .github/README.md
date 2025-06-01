# 🔍 Korean Text De-obfuscation Fine-tuning Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)

이 프로젝트는 **Naver HyperCLOVAX-SEED-Text-Instruct-0.5B** 모델을 한국어 텍스트 비난독화(De-obfuscation) 작업을 위해 fine-tuning하는 연구 프로젝트입니다.

## 📋 프로젝트 개요

### 팀 정보
- **팀명**: 91veMe4Plus
- **프로젝트명**: 한국어 텍스트 비난독화 AI 모델 성능 분석
- **라이선스**: MIT License
- **저작권**: Copyright (c) 2025 91veMe4Plus

### 목표
- 난독화된 한국어 텍스트를 원본 텍스트로 복원하는 모델 개발
- **LoRA (Low-Rank Adaptation)**를 사용한 효율적인 fine-tuning
- 다양한 텍스트 유형에 대한 성능 평가 및 비교
- 하이퍼파라미터 최적화를 통한 모델 성능 분석

## 🎯 핵심 목표

1. **모델 파인튜닝**: Naver HyperCLOVAX-SEED-Text-Instruct-0.5B 모델을 한국어 텍스트 비난독화 작업에 최적화
2. **하이퍼파라미터 최적화**: Learning Rate, Batch Size, Dataset Size 등 다양한 파라미터의 영향 분석
3. **성능 비교 분석**: 원본 모델 대비 파인튜닝 모델의 정량적/정성적 성능 개선 측정
4. **효율적 학습 기법**: LoRA (Low-Rank Adaptation)를 활용한 파라미터 효율적 파인튜닝

## 📊 데이터셋 정보

### 훈련 데이터셋
6가지 한국어 텍스트 유형의 난독화 데이터 (총 696,024 샘플):
- `구어체_대화체_16878_sample_난독화결과.csv` (16,878 샘플)
- `뉴스문어체_281932_sample_난독화결과.csv` (281,932 샘플)
- `문화문어체_25628_sample_난독화결과.csv` (25,628 샘플)
- `전문분야 문어체_306542_sample_난독화결과.csv` (306,542 샘플)
- `조례문어체_36339_sample_난독화결과.csv` (36,339 샘플)
- `지자체웹사이트 문어체_28705_sample_난독화결과.csv` (28,705 샘플)

### 테스트 데이터셋
- `testdata.csv` (1,002 샘플)
- 난독화된 텍스트와 원본 텍스트 쌍으로 구성

## 🔧 파인튜닝 전략

### 1. 모델 아키텍처 및 설정

#### 베이스 모델
- **모델명**: Naver HyperCLOVAX-SEED-Text-Instruct-0.5B
- **모델 유형**: Causal Language Model (Auto-regressive)
- **파라미터 수**: 0.5B (5억 개)
- **토크나이저**: HyperCLOVAX 전용 토크나이저

#### 양자화 설정 (메모리 최적화)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### 2. LoRA (Low-Rank Adaptation) 구성

#### LoRA 하이퍼파라미터
- **Rank (r)**: 16
- **Alpha**: 32 (scaling parameter)
- **Dropout**: 0.1
- **Target Modules**: 
  - `q_proj`, `k_proj`, `v_proj`, `o_proj` (Attention layers)
  - `gate_proj`, `up_proj`, `down_proj` (Feed-forward layers)
- **Task Type**: CAUSAL_LM
- **Bias**: "none"

#### LoRA 설정 코드
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### 3. 데이터 전처리 전략

#### 프롬프트 템플릿
```
### 지시사항:
다음 난독화된 한국어 텍스트를 원래 텍스트로 복원해주세요.

난독화된 텍스트: {obfuscated_text}

### 응답:
{original_text}
```

#### 데이터 샘플링 전략
- **균형 샘플링**: 6가지 텍스트 유형에서 균등하게 샘플링
- **최대 길이**: 512 토큰으로 제한
- **훈련/검증 분할**: 9:1 비율

### 4. 훈련 설정

#### 공통 훈련 파라미터
- **Epochs**: 3
- **Gradient Accumulation Steps**: 4
- **Warmup Steps**: 100
- **Weight Decay**: 0.01
- **FP16**: True (Mixed Precision Training)
- **Evaluation Strategy**: Steps (매 200 스텝)
- **Save Strategy**: Best model 기준 저장

#### 최적화 알고리즘
- **Optimizer**: AdamW
- **Scheduler**: Linear Warmup + Decay

### 5. 추론 설정

#### 생성 파라미터
- **Max New Tokens**: 128
- **Do Sample**: True
- **Temperature**: 0.7
- **Top-p**: 0.9
- **Repetition Penalty**: 미적용

### 6. 성능 최적화 기법

#### 메모리 최적화
- **4-bit 양자화**: BitsAndBytes 활용
- **Gradient Checkpointing**: 메모리 사용량 감소
- **DataLoader Pin Memory**: False (Colab 환경 최적화)

#### 훈련 안정성
- **Learning Rate Warmup**: 초기 100 스텝 동안 점진적 증가
- **Gradient Clipping**: 기본값 적용
- **Early Stopping**: Validation Loss 기준

### 7. 실험별 변수 설정

#### Learning Rate 실험
- **실험 A**: 1e-4 (보수적 학습)
- **실험 B**: 5e-4 (적극적 학습)

#### Batch Size 실험  
- **실험 A**: Per Device Batch Size 1 (메모리 효율)
- **실험 B**: Per Device Batch Size 2 (균형)
- **실험 C**: Per Device Batch Size 4 (속도 우선)

#### Dataset Size 실험
- **실험 A**: 10,000 샘플 (효율성 검증)
- **실험 B**: 30,000 샘플 (성능 최대화)

## 🧪 실험 설계 및 분석

### 1. Learning Rate 실험
**파일**: `learning_rate_hyperclova_deobfuscation_finetuning.ipynb`

#### 실험 조건
- **실험 A**: 낮은 학습률 (1e-4)
- **실험 B**: 높은 학습률 (5e-4)

#### 결과 요약
| 모델 | BLEU 점수 | ROUGE-1 | ROUGE-2 | ROUGE-L | 문자 정확도 | 추론 시간 |
|------|-----------|---------|---------|---------|-------------|----------|
| 원본 모델 | 0.0029 | 0.138 | 0.064 | 0.138 | 0.151 | 6.78s |
| 1e-4 Learning Rate | 0.0233 | 0.276 | 0.148 | 0.279 | 0.332 | 3.03s |
| 5e-4 Learning Rate | 0.0211 | 0.277 | 0.148 | 0.279 | 0.313 | 2.89s |

### 2. Batch Size 실험
**파일**: `batch_size_hyperclova_deobfuscation_finetuning.ipynb`

#### 실험 조건
- **Batch Size 1**: 메모리 효율적, 학습 안정성 높음
- **Batch Size 2**: 균형잡힌 설정
- **Batch Size 4**: 빠른 수렴, 높은 메모리 사용

#### 결과 요약
| 모델 | BLEU 점수 | ROUGE-1 | ROUGE-2 | ROUGE-L | 문자 정확도 | 추론 시간 |
|------|-----------|---------|---------|---------|-------------|----------|
| 원본 모델 | 0.0022 | 0.123 | 0.052 | 0.122 | 0.145 | 7.37s |
| 배치 크기 1 | 0.0193 | 0.279 | 0.145 | 0.279 | 0.315 | 3.12s |
| 배치 크기 2 | 0.0192 | 0.279 | 0.148 | 0.279 | 0.326 | 3.12s |
| 배치 크기 4 | 0.0220 | 0.279 | 0.149 | 0.279 | 0.331 | 3.15s |

### 3. Dataset Size 실험
**파일**: `datasets_hyperclova_deobfuscation_finetuning.ipynb`

#### 실험 조건
- **실험 A**: 1만개 샘플로 파인튜닝
- **실험 B**: 3만개 샘플로 파인튜닝

#### 결과 요약
| 모델 | BLEU 점수 | ROUGE-1 | ROUGE-2 | ROUGE-L | 문자 정확도 | 추론 시간 |
|------|-----------|---------|---------|---------|-------------|----------|
| 원본 모델 | 0.0024 | 0.124 | 0.063 | 0.123 | 0.124 | 7.53s |
| 10K 데이터셋 | 0.0220 | 0.279 | 0.148 | 0.278 | 0.327 | 3.13s |
| 30K 데이터셋 | 0.0201 | 0.279 | 0.149 | 0.279 | 0.324 | 3.12s |

## 📈 성능 분석 결과

### 주요 성과
1. **파인튜닝 효과 입증**: 모든 실험에서 원본 모델 대비 상당한 성능 향상 확인
   - BLEU 점수: 7-10배 향상
   - ROUGE 점수: 2배 이상 향상
   - 문자 정확도: 2배 이상 향상
   - 추론 시간: 50% 이상 단축

2. **최적 하이퍼파라미터 발견**:
   - Learning Rate: 1e-4가 약간 더 나은 성능
   - Batch Size: 4가 가장 균형잡힌 성능
   - Dataset Size: 10K 데이터셋이 효율적인 성능

## 🔧 기술 스택

### 핵심 라이브러리
- **Transformers**: HuggingFace 트랜스포머 모델 라이브러리
- **PEFT**: Parameter Efficient Fine-Tuning (LoRA 구현)
- **TRL**: Transformer Reinforcement Learning
- **Datasets**: HuggingFace 데이터셋 라이브러리
- **BitsAndBytes**: 양자화 라이브러리
- **Accelerate**: 분산 학습 지원

### 평가 메트릭
- **BLEU Score**: 기계번역 품질 평가
- **ROUGE-1/2/L**: 요약 품질 평가
- **문자 정확도**: 문자 단위 정확도
- **완전 일치율**: 전체 텍스트 완전 일치 비율
- **추론 시간**: 모델 추론 속도

## 📁 프로젝트 구조

```
FineTuningLLM/
├── 📄 학습 노트북
│   ├── learning_rate_hyperclova_deobfuscation_finetuning.ipynb
│   ├── batch_size_hyperclova_deobfuscation_finetuning.ipynb
│   └── datasets_hyperclova_deobfuscation_finetuning.ipynb
│
├── 📊 성능 분석 노트북
│   ├── model_performance_analysis_learning_rate.ipynb
│   ├── model_performance_analysis_batch_size.ipynb
│   ├── model_performance_analysis_datasets.ipynb
│   └── team_all_model_performance_analysis_learning_rate.ipynb
│
├── 📂 데이터셋
│   ├── testdata.csv
│   └── [6개의 한국어 텍스트 유형별 난독화 데이터]
│
├── 🤖 훈련된 모델
│   ├── hyperclova-deobfuscation-lora-1e-4-learning-rate/
│   ├── hyperclova-deobfuscation-lora-5e-4-learning-rate/
│   ├── hyperclova-deobfuscation-lora-with-1-batch-size/
│   ├── hyperclova-deobfuscation-lora-with-2-batch-size/
│   ├── hyperclova-deobfuscation-lora-with-4-batch-size/
│   ├── hyperclova-deobfuscation-lora-with-10k-datasets/
│   └── hyperclova-deobfuscation-lora-with-30k-datasets/
│
├── 📈 분석 결과
│   ├── 1차 학습률 조정 차이에 대한 성능 분석/
│   ├── 1차 배치값 조정 차이에 대한 성능 분석/
│   └── 1차 학습량에 차이에 따른 성능 분석/
│
└── 📋 문서
    ├── LICENSE
    ├── .github/README.md
    └── PROJECT_DOCUMENTATION.md
```

## 🛠️ 설치 및 실행 가이드

### 필수 요구사항
- Python 3.8+
- CUDA 지원 GPU (권장)
- 16GB+ RAM
- 충분한 저장 공간 (모델 및 데이터용)

### 설치 방법
```bash
# 저장소 클론
git clone https://github.com/91veMe4Plus/FineTuningLLM.git
cd FineTuningLLM

# 필수 패키지 설치
pip install transformers>=4.35.0
pip install peft>=0.6.0
pip install trl>=0.7.0
pip install datasets>=2.14.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.24.0
pip install evaluate>=0.4.0
pip install rouge-score>=0.1.2
pip install gradio>=4.0.0
pip install scikit-learn>=1.3.0
```

### 실행 방법
1. **데이터셋 준비**: 데이터셋을 적절한 경로에 배치
2. **실험 실행**: 원하는 실험 노트북을 선택하여 실행
   ```bash
   jupyter notebook learning_rate_hyperclova_deobfuscation_finetuning.ipynb
   ```
3. **성능 분석**: 성능 분석 노트북으로 결과 분석
   ```bash
   jupyter notebook model_performance_analysis_learning_rate.ipynb
   ```

## 🔮 향후 연구 방향

### 1. 모델 확장
- 더 큰 모델 (1B, 3B 파라미터)에서의 실험
- 다른 베이스 모델 (KoBART, KoGPT 등) 비교 분석

### 2. 데이터 확장
- 더 다양한 텍스트 도메인 추가
- 더 큰 데이터셋 (50K, 100K) 실험
- 실시간 데이터 수집 및 지속 학습

### 3. 기술 개선
- 다른 PEFT 기법 (AdaLoRA, QLoRA 등) 비교
- 앙상블 모델 구축
- 실시간 추론 API 개발

### 4. 응용 분야 확장
- 다국어 비난독화 확장
- 실시간 채팅 필터링 시스템
- 웹 브라우저 확장 프로그램

## 📊 시각화 및 분석 결과

프로젝트에는 다음과 같은 시각화 결과물이 포함되어 있습니다:

### CSV 분석 파일
- `model_performance_summary_*.csv`: 모델별 성능 요약
- `detailed_model_comparison_*.csv`: 상세 비교 분석 결과
- `finetuning_effect_analysis.csv`: 파인튜닝 효과 분석

### 시각화 이미지
- 성능 비교 차트 (PNG 형식)
- 문자 정확도 분포 히스토그램
- 추론 시간 비교 그래프
- 텍스트 길이별 성능 분석
- 카테고리별 정확도 분포

---

**© 2025 91veMe4Plus Team. All rights reserved.**
