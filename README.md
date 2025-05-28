# 🚀 HyperCLOVAX Korean Text De-obfuscation Fine-tuning Project

이 프로젝트는 **Naver HyperCLOVAX-SEED-Text-Instruct-0.5B** 모델을 한국어 텍스트 비난독화(De-obfuscation) 작업을 위해 fine-tuning하는 연구 프로젝트입니다.

## 📋 프로젝트 개요

### 목표
- 난독화된 한국어 텍스트를 원본 텍스트로 복원하는 모델 개발
- **LoRA (Low-Rank Adaptation)**를 사용한 효율적인 fine-tuning
- 다양한 텍스트 유형에 대한 성능 평가 및 비교
- 데이터셋 크기에 따른 모델 성능 분석

### 주요 특징
- 🎯 **Target Model**: HyperCLOVAX-SEED-Text-Instruct-0.5B
- 🔧 **Training Method**: LoRA Fine-tuning
- 📊 **Dataset Sizes**: 10K, 30K 샘플
- 📈 **Performance Metrics**: BLEU, ROUGE, Character Accuracy
- 🗂️ **Text Categories**: 구어체, 뉴스, 문화, 전문분야, 조례, 지자체웹사이트

## 📁 프로젝트 구조

```
FineTuningLLM/
├── 📓 hyperclova_deobfuscation_finetuning.ipynb    # 메인 fine-tuning 노트북
├── 📊 model_performance_analysis.ipynb             # 모델 성능 비교 분석
├── 📄 testdata.csv                                 # 테스트 데이터 (1,002 샘플)
├── 📊 구어체_대화체_16878_sample_난독화결과.csv          # 구어체 훈련 데이터
├── 📊 뉴스문어체_281932_sample_난독화결과.csv           # 뉴스 훈련 데이터
├── 📊 문화문어체_25628_sample_난독화결과.csv            # 문화 훈련 데이터
├── 📊 전문분야 문어체_306542_sample_난독화결과.csv        # 전문분야 훈련 데이터
├── 📊 조례문어체_36339_sample_난독화결과.csv            # 조례 훈련 데이터
├── 📊 지자체웹사이트 문어체_28705_sample_난독화결과.csv    # 지자체웹사이트 훈련 데이터
├── 🤖 hyperclova-deobfuscation-lora-with-10k-datasets/  # 10K 모델
└── 🤖 hyperclova-deobfuscation-lora-with-30k-datasets/  # 30K 모델
```

## 🛠️ 환경 설정

### 시스템 요구사항
- **GPU**: NVIDIA GPU (8GB VRAM 이상 권장)
- **Python**: 3.8+
- **CUDA**: 11.8+
- **Environment**: Google Colab 또는 로컬 환경

### 필수 패키지 설치

```bash
pip install transformers>=4.35.0
pip install peft>=0.6.0
pip install trl>=0.7.0
pip install datasets>=2.14.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.24.0
pip install evaluate>=0.4.0
pip install rouge-score>=0.1.2
pip install sentencepiece>=0.1.99
pip install scikit-learn>=1.3.0
pip install matplotlib seaborn plotly
```

## 📚 데이터셋

### 훈련 데이터
총 6개 카테고리의 한국어 텍스트 데이터:

| 카테고리 | 샘플 수 | 설명 |
|---------|---------|------|
| 구어체 대화체 | 16,878 | 일상 대화 및 구어체 텍스트 |
| 뉴스 문어체 | 281,932 | 뉴스 기사 및 보도 자료 |
| 문화 문어체 | 25,628 | 문화 관련 텍스트 |
| 전문분야 문어체 | 306,542 | 전문 분야 문서 |
| 조례 문어체 | 36,339 | 법률 및 조례 문서 |
| 지자체웹사이트 문어체 | 28,705 | 공공기관 웹사이트 텍스트 |

### 테스트 데이터
- **파일**: `testdata.csv`
- **샘플 수**: 1,002개
- **구조**: `index`, `original`, `obfuscated` 컬럼

## 🚀 사용법

### 1. Fine-tuning 실행

```python
# hyperclova_deobfuscation_finetuning.ipynb 실행
# - 데이터 로딩 및 전처리
# - LoRA 설정 및 모델 훈련
# - 체크포인트 저장
```

### 2. 모델 성능 분석

```python
# model_performance_analysis.ipynb 실행
# - 원본 모델 vs Fine-tuned 모델 비교
# - 정량적/정성적 성능 평가
# - 결과 시각화
```

### 3. 추론 예시

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 베이스 모델 로드
model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(model, "./hyperclova-deobfuscation-lora-with-30k-datasets")

# 추론
obfuscated_text = "별 한 게토 았깝땀. 왜 싸람듯릭 펼 1캐를 쥰눈징..."
prompt = f"다음 난독화된 텍스트를 원래대로 복원해주세요.\n\n난독화된 텍스트: {obfuscated_text}\n\n복원된 텍스트:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📊 성능 결과

### 정량적 평가 지표

| 모델 | BLEU Score | ROUGE-L | Character Accuracy |
|------|------------|---------|-------------------|
| 원본 모델 | - | - | - |
| 10K Fine-tuned | - | - | - |
| 30K Fine-tuned | - | - | - |

### 주요 발견사항
- **데이터셋 크기**: 30K > 10K 데이터셋으로 훈련된 모델이 더 우수한 성능
- **카테고리별 성능**: 뉴스 문어체와 전문분야 문어체에서 높은 성능
- **훈련 효율성**: LoRA 방법으로 효율적인 fine-tuning 달성

## 🔧 모델 아키텍처

### LoRA 설정
```python
lora_config = LoraConfig(
    r=64,                    # rank
    lora_alpha=16,          # scaling parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### 훈련 파라미터
- **Learning Rate**: 2e-4
- **Batch Size**: 4 (gradient accumulation steps: 4)
- **Max Length**: 512 tokens
- **Epochs**: 3
- **Optimizer**: AdamW
- **Scheduler**: Linear with warmup

## 📈 실험 결과

### 훈련 진행상황
- **10K 모델**: 1,686 steps (최종 체크포인트)
- **30K 모델**: 5,061 steps (최종 체크포인트)
- **중간 체크포인트**: 각 모델별 다수 저장

### 성능 비교
상세한 성능 분석 결과는 `model_performance_analysis.ipynb`에서 확인할 수 있습니다.

## 🔮 향후 계획

- [ ] 더 큰 데이터셋 (50K, 100K)으로 확장 실험
- [ ] 다른 LoRA 설정 (rank, alpha) 최적화
- [ ] 다양한 베이스 모델 비교 실험
- [ ] 실시간 추론 API 개발
- [ ] 웹 인터페이스 구축

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 라이센스

이 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 라이센스를 따릅니다.

## 🙏 감사의 말

- **Naver HyperCLOVAX** 팀의 우수한 베이스 모델 제공
- **Hugging Face** 커뮤니티의 Transformers 및 PEFT 라이브러리
- **Google Colab**의 무료 GPU 환경 제공

## 📞 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

*이 프로젝트는 한국어 텍스트 비난독화 연구를 위한 교육 및 연구 목적으로 개발되었습니다.*
