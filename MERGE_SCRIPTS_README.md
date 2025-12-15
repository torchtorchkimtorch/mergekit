# Model Merge Scripts 사용 가이드

이 디렉토리에는 DELLA, DARE, TIES 세 가지 모델 병합 방법을 실행하기 위한 스크립트가 포함되어 있습니다.

## 📁 파일 구조

```
.
├── run_merges.sh       # 세 가지 방법 모두 실행 (통합 스크립트)
├── run_della.sh        # DELLA만 실행
├── run_dare.sh         # DARE만 실행
├── run_ties.sh         # TIES만 실행
├── wbl_della.yaml      # DELLA 설정 파일
├── wbl_dare.yaml       # DARE 설정 파일
└── wbl_ties.yaml       # TIES 설정 파일
```

---

## 🚀 빠른 시작

### 1. 모든 병합 방법 실행 (권장)

```bash
./run_merges.sh
```

이 명령은 DELLA, DARE, TIES 세 가지 방법을 순차적으로 실행합니다.

### 2. 개별 병합 방법 실행

```bash
# DELLA만 실행
./run_della.sh

# DARE만 실행
./run_dare.sh

# TIES만 실행
./run_ties.sh
```

---

## ⚙️ 환경 변수 설정

스크립트 실행 전에 환경 변수로 옵션을 커스터마이징할 수 있습니다:

### 기본 사용법 (기본값 사용)

```bash
./run_merges.sh
```

### 커스텀 설정

```bash
# GPU 선택
CUDA_VISIBLE_DEVICES=0 ./run_merges.sh

# 출력 디렉토리 변경
OUTPUT_DIR="/path/to/output" ./run_della.sh

# CUDA 비활성화 (CPU만 사용)
USE_CUDA=false ./run_dare.sh

# 낮은 CPU 메모리 모드 활성화 (VRAM이 많을 때)
LOW_CPU_MEMORY=true ./run_ties.sh

# 여러 옵션 조합
CUDA_VISIBLE_DEVICES=1 \
OUTPUT_DIR="./my_models" \
LOW_CPU_MEMORY=true \
./run_merges.sh
```

---

## 🔧 환경 변수 상세 설명

| 환경 변수 | 기본값 | 설명 |
|----------|--------|------|
| `CUDA_VISIBLE_DEVICES` | `0` | 사용할 GPU 번호 |
| `OUTPUT_DIR` | `./merged_models` | 병합된 모델 저장 경로 |
| `USE_CUDA` | `true` | GPU 사용 여부 (`--cuda` 플래그) |
| `LAZY_UNPICKLE` | `true` | 메모리 효율적 로딩 (`--lazy-unpickle`) |
| `LOW_CPU_MEMORY` | `false` | GPU에 중간값 저장 (`--low-cpu-memory`) |
| `TRUST_REMOTE_CODE` | `true` | 원격 코드 신뢰 (`--trust-remote-code`) |

---

## 📂 출력 구조

기본 출력 디렉토리: `./merged_models/`

```
merged_models/
├── della-merged/          # DELLA 병합 결과
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── dare-merged/           # DARE 병합 결과
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── ties-merged/           # TIES 병합 결과
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

로그 파일: `./logs/merge_YYYYMMDD_HHMMSS.log`

---

## 📊 병합 방법 비교

| 방법 | 특징 | 파라미터 |
|------|------|----------|
| **DELLA** | Magnitude-based adaptive pruning | `density: 0.2`, `epsilon: 0.15` |
| **DARE** | Random pruning + rescaling | `density: 0.2`, `rescale: true` |
| **TIES** | Sign consensus + magnitude pruning | `density: 0.2`, `normalize: true` |

---

## 💡 사용 예시

### 예시 1: 기본 실행

```bash
./run_merges.sh
```

### 예시 2: GPU 1번 사용, 커스텀 출력 경로

```bash
CUDA_VISIBLE_DEVICES=1 OUTPUT_DIR="/data/merged_models" ./run_merges.sh
```

### 예시 3: CPU만 사용 (GPU 없을 때)

```bash
USE_CUDA=false ./run_della.sh
```

### 예시 4: 메모리 최적화 (VRAM > RAM인 경우)

```bash
LOW_CPU_MEMORY=true LAZY_UNPICKLE=true ./run_dare.sh
```

### 예시 5: 특정 방법만 실행

```bash
# DELLA만 실행하고 결과를 특정 경로에 저장
OUTPUT_DIR="./results/della_experiment_1" ./run_della.sh
```

---

## 🔍 로그 확인

모든 실행 로그는 `./logs/` 디렉토리에 타임스탬프와 함께 저장됩니다:

```bash
# 최신 로그 확인
tail -f ./logs/merge_*.log

# 특정 로그 확인
cat ./logs/merge_20231215_143022.log
```

---

## ⚠️ 주의사항

1. **디스크 공간**: 각 병합 결과는 약 26GB (13B 모델 기준)의 공간을 차지합니다
2. **메모리**: 최소 32GB RAM 권장 (GPU 사용 시 16GB VRAM 권장)
3. **시간**: 각 병합은 GPU 사용 시 약 30분~2시간 소요됩니다
4. **모델 다운로드**: 처음 실행 시 HuggingFace에서 모델을 자동으로 다운로드합니다

---

## 🛠️ 문제 해결

### CUDA Out of Memory 에러

```bash
# 낮은 CPU 메모리 모드 활성화
LOW_CPU_MEMORY=true ./run_merges.sh
```

### 모델 다운로드 실패

```bash
# HuggingFace 토큰 설정
export HF_TOKEN="your_token_here"
./run_merges.sh
```

### 권한 에러

```bash
# 실행 권한 부여
chmod +x run_*.sh
```

---

## 📝 설정 파일 수정

각 YAML 파일을 수정하여 병합 파라미터를 조정할 수 있습니다:

```yaml
# wbl_della.yaml 예시
parameters:
  density: 0.2      # 유지할 파라미터 비율 (0.1 ~ 0.5)
  epsilon: 0.15     # DELLA adaptive range (0.1 ~ 0.2)
  normalize: true   # 정규화 여부
```

---

## 📚 추가 정보

- **DELLA 논문**: Depth Low-Rank Approximation
- **DARE 논문**: Drop And REscale (p = 1 - density)
- **TIES 논문**: Trim, Elect Sign & Merge

각 방법의 상세한 설명은 `docs/merge_methods.md`를 참고하세요.

