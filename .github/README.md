# HyperCLOVAX 한국어 텍스트 비난독화 파인튜닝

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow.svg)](https://huggingface.co/transformers)

이 프로젝트는 Naver의 HyperCLOVAX-SEED-Text-Instruct-0.5B 모델을 한국어 텍스트 비난독화를 위해 LoRA(Low-Rank Adaptation)를 사용하여 효율적으로 파인튜닝하는 방법을 보여줍니다.

## 🎯 프로젝트 개요

이 프로젝트의 목표는 난독화된 한국어 텍스트를 원본 형태로 복원할 수 있는 모델을 훈련시키는 것입니다. 다음과 같은 다양한 유형의 한국어 텍스트로 훈련됩니다:

- **구어체 (구어체)** - 16,878개 샘플
- **뉴스문어체 (뉴스 텍스트)** - 281,932개 샘플  
- **문화문어체 (문화 텍스트)** - 25,628개 샘플
- **전문분야문어체 (전문 기술 텍스트)** - 306,542개 샘플
- **조례문어체 (법령 텍스트)** - 36,339개 샘플
- **지자체웹사이트문어체 (지자체 웹사이트 텍스트)** - 28,705개 샘플

## 🚀 주요 기능

- **효율적인 파인튜닝**: 메모리 효율적인 훈련을 위한 LoRA(Low-Rank Adaptation) 사용
- **4비트 양자화**: 메모리 사용량 감소를 위한 BitsAndBytesConfig 구현
- **다중 카테고리 지원**: 다양한 한국어 텍스트 유형 처리
- **인터랙티브 데모**: 실시간 비난독화를 위한 Gradio 인터페이스
- **종합적인 평가**: BLEU 및 ROUGE 점수 메트릭
- **GPU 최적화**: 자동 디바이스 매핑을 통한 CUDA 지원

## 📋 요구사항

### 하드웨어 요구사항
- **GPU**: CUDA 지원 NVIDIA GPU (권장: 16GB+ VRAM)
- **메모리**: 대용량 데이터셋을 위해 32GB+ RAM 권장
- **저장공간**: 모델 및 데이터셋을 위한 50GB+ 여유 공간

### 소프트웨어 요구사항
- Python 3.8+
- CUDA 11.8+ (GPU 가속을 위해)
- 상세한 패키지 의존성은 `requirements.txt` 참조

### 주요 의존성
```
transformers
peft
trl
datasets
bitsandbytes
accelerate
torch>=2.0.0
gradio
evaluate
rouge-score
sentencepiece
protobuf
scikit-learn
pandas
numpy
```

## 🛠️ 설치

1. **저장소 클론**:
```bash
git clone <repository-url>
cd FineTuningLLM
```

2. **가상환경 생성**:
```bash
python -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
```

3. **의존성 설치**:
```bash
pip install transformers peft trl datasets bitsandbytes accelerate evaluate rouge-score sentencepiece protobuf gradio scikit-learn pandas numpy torch
```

4. **CUDA 설치 확인** (선택사항):
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 데이터셋 구조

프로젝트는 난독화된 한국어 텍스트와 원본 텍스트가 포함된 CSV 파일을 사용합니다:

```
데이터셋 파일:
├── 구어체_대화체_16878_sample_난독화결과.csv
├── 뉴스문어체_281932_sample_난독화결과.csv
├── 문화문어체_25628_sample_난독화결과.csv
├── 전문분야 문어체_306542_sample_난독화결과.csv
├── 조례문어체_36339_sample_난독화결과.csv
└── 지자체웹사이트 문어체_28705_sample_난독화결과.csv
```

각 CSV 파일의 구성:
- `obfuscated`: 난독화된 한국어 텍스트
- `original`: 원본 한국어 텍스트
- 추가 메타데이터 컬럼들

### 로컬 파일 경로 설정

노트북을 로컬에서 실행할 때는 다음과 같이 파일 경로를 수정하세요:

```python
# 로컬 실행용 파일 경로
data_files = {
    '구어체_대화체': './구어체_대화체_16878_sample_난독화결과.csv',
    '뉴스문어체': './뉴스문어체_281932_sample_난독화결과.csv',
    '문화문어체': './문화문어체_25628_sample_난독화결과.csv',
    '전문분야문어체': './전문분야 문어체_306542_sample_난독화결과.csv',
    '조례문어체': './조례문어체_36339_sample_난독화결과.csv',
    '지자체웹사이트문어체': './지자체웹사이트 문어체_28705_sample_난독화결과.csv'
}
```

## 🏃‍♂️ 빠른 시작

### 노트북 실행

1. **Jupyter Notebook 실행**:
```bash
jupyter notebook hyperclova_deobfuscation_finetuning.ipynb
```

2. **또는 Jupyter Lab 사용**:
```bash
jupyter lab hyperclova_deobfuscation_finetuning.ipynb
```

3. **Google Colab 사용**:
   - 노트북을 Google Colab에 업로드
   - 데이터셋 접근을 위해 Google Drive 마운트
   - 노트북의 데이터 로딩 섹션에서 파일 경로 설정

### 훈련된 모델 사용

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
)

# 파인튜닝된 LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./hyperclova-deobfuscation-lora")
tokenizer = AutoTokenizer.from_pretrained("./hyperclova-deobfuscation-lora")

def deobfuscate_text(obfuscated_text):
    prompt = f"### 지시사항:\n다음 난독화된 한국어 텍스트를 원래 텍스트로 복원해주세요.\n\n난독화된 텍스트: {obfuscated_text}\n\n### 응답:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### 응답:")[1].strip()

# 사용 예시
obfuscated = "안녀하쎼요, 반갑쏘니댜!"
original = deobfuscate_text(obfuscated)
print(f"복원된 텍스트: {original}")
```

## 🔧 모델 설정

### 데이터 전처리
노트북에서는 메모리 효율성을 위해 10,000개 샘플로 제한하여 훈련합니다:

```python
# 메모리 절약을 위한 샘플 크기 조정
sample_size = 10000  # 필요에 따라 조정 가능
instruction_df = create_instruction_dataset(combined_df, sample_size)
```

### LoRA 설정
```python
lora_config = LoraConfig(
    r=16,                    # 랭크
    lora_alpha=32,          # 스케일링 파라미터
    target_modules=[        # 타겟 어텐션 레이어
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### 훈련 설정
```python
training_args = TrainingArguments(
    output_dir="./hyperclova-deobfuscation-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    eval_strategy="steps",  # 업데이트된 파라미터명
    eval_steps=200,
    save_steps=200,
    logging_steps=10
)
```

## 📈 성능 메트릭

모델은 다음과 같은 여러 메트릭으로 평가됩니다:

- **BLEU Score**: 예측된 텍스트와 참조 텍스트 간의 n-gram 겹침 측정
- **ROUGE-1/2/L**: 텍스트 요약을 위한 recall 지향적 이해도 평가
- **정성적 분석**: 비난독화 품질의 수동 검사

### 평가 예시
노트북에는 100개 샘플에 대한 정량적 평가가 포함되어 있습니다:

```python
# 100개 샘플로 평가
eval_samples = val_df.sample(n=100, random_state=42)
# BLEU, ROUGE 스코어 계산
metrics = calculate_metrics(predictions, references)
```

### 예시 결과
```
BLEU Score: 0.xxxx
ROUGE-1: 0.xxxx
ROUGE-2: 0.xxxx
ROUGE-L: 0.xxxx
```

## 🎮 인터랙티브 데모

노트북에는 실시간 텍스트 비난독화를 위한 Gradio 기반 웹 인터페이스가 포함되어 있습니다:

```python
import gradio as gr

demo = gr.Interface(
    fn=deobfuscate_interface,
    inputs=gr.Textbox(label="난독화된 한국어 텍스트"),
    outputs=gr.Textbox(label="복원된 원본 텍스트"),
    title="HyperCLOVAX 한국어 텍스트 비난독화",
    examples=[
        ["안녀하쎼요, 반갑쏘니댜!"],
        ["오늬 날씨갸 맆이 좆네욘."],
        ["한큿어 쳬연어 처륄예 댕햔 연귝을 해보갰습닏댜."]
    ]
)

demo.launch(share=True)
```

## 🐍 환경 호환성

### Google Colab
노트북은 Google Colab에서 실행되도록 최적화되어 있습니다:
- Google Drive 마운트 지원
- GPU 메모리 최적화
- 자동 패키지 설치

### 로컬 환경
로컬에서 실행하려면:
1. Google Colab 관련 코드 주석 처리
2. 데이터 파일 경로를 로컬 경로로 수정
3. 필요한 패키지 사전 설치

## 📁 프로젝트 구조

```
FineTuningLLM/
├── .github/
│   └── README.md                           # 이 파일
├── hyperclova_deobfuscation_finetuning.ipynb  # 메인 노트북
├── LICENSE                                # 프로젝트 라이선스
├── 구어체_대화체_16878_sample_난독화결과.csv      # 구어체 데이터셋
├── 뉴스문어체_281932_sample_난독화결과.csv        # 뉴스 데이터셋
├── 문화문어체_25628_sample_난독화결과.csv         # 문화 데이터셋
├── 전문분야 문어체_306542_sample_난독화결과.csv    # 전문분야 데이터셋
├── 조례문어체_36339_sample_난독화결과.csv         # 조례 데이터셋
└── 지자체웹사이트 문어체_28705_sample_난독화결과.csv # 지자체 웹사이트 데이터셋
```

## ⚙️ 실행 시 주의사항

### 메모리 최적화
- 노트북에서는 `sample_size=10000`으로 제한하여 메모리 사용량 절약
- 더 많은 데이터 사용 시 GPU 메모리와 RAM 용량 확인 필요
- 4비트 양자화(`BitsAndBytesConfig`) 사용으로 메모리 효율성 향상

### 파일 경로 설정
**Google Colab 사용 시:**
```python
# Google Drive 경로 (노트북 기본값)
'/content/drive/MyDrive/파일명.csv'
```

**로컬 실행 시:**
```python
# 로컬 경로로 수정 필요
'./파일명.csv'
```

### GPU 요구사항
- NVIDIA GPU 권장 (8GB+ VRAM)
- CUDA 11.8+ 설치 필요
- CPU만으로도 실행 가능하지만 속도 현저히 저하

## 🤝 기여하기

기여는 언제나 환영합니다! Pull Request를 자유롭게 제출해 주세요. 주요 변경사항의 경우, 먼저 issue를 열어 논의해 주시기 바랍니다.

### 개발 환경 설정
1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 열기

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다 - 자세한 내용은 [LICENSE](../LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **Naver AI**: HyperCLOVAX-SEED-Text-Instruct-0.5B 베이스 모델 제공
- **Hugging Face**: transformers 라이브러리와 모델 호스팅
- **Microsoft**: PEFT 라이브러리의 LoRA 구현
- **한국어 NLP 커뮤니티**: 데이터셋 기여 및 연구

## 📞 지원

문제가 발생하거나 질문이 있으시면:

1. 기존 해결책을 위해 [Issues](../../issues) 페이지 확인
2. 문제에 대한 자세한 정보와 함께 새 issue 생성
3. 버그 신고 시 시스템 사양과 오류 로그 포함

## 🔗 관련 자료

- [HyperCLOVAX 모델 카드](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)
- [한국어 언어처리 자료](https://github.com/topics/korean-nlp)

---

**참고**: 이 프로젝트는 연구 및 교육 목적입니다. 이 코드를 사용할 때 모델 사용 약관 및 데이터 개인정보보호 규정을 준수하시기 바랍니다.
