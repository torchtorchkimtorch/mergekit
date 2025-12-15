# mergekit

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/arcee-ai/mergekit/pre-commit.yml?label=Tests)](https://github.com/arcee-ai/mergekit/actions/workflows/pre-commit.yml)
[![Arcee Discord](https://img.shields.io/badge/Arcee%20Discord-Arcee%20Discord?logo=discord\&logoColor=white\&color=5865F2)](https://discord.gg/arceeai)

`mergekit`은 사전 학습된 언어 모델을 **병합(merge)** 하기 위한 툴킷입니다. `mergekit`은 **out-of-core 방식**을 사용하여, 제한된 자원 환경에서도 비교적 복잡한 병합을 수행할 수 있도록 설계되었습니다. 병합은 **CPU만으로도 실행 가능**하며, **최소 8GB VRAM**만으로도 GPU 가속이 가능합니다. 다양한 병합 알고리즘을 지원하며, 앞으로도 지속적으로 추가될 예정입니다.

---

## Contents

* [Why Merge Models?](#why-merge-models)
* [Features](#features)
* [Installation](#installation)
* [Community & Support](#community--support)

  * [Contributing](#contributing)
  * [Community Tools](#community-tools)
* [Usage](#usage)
* [Merge Configuration](#merge-configuration)

  * [Parameter Specification](#parameter-specification)
  * [Tokenizer Configuration](#tokenizer-configuration)
  * [Chat Template Configuration](#chat-template-configuration)
  * [Examples](#examples)
* [Merge Methods](#merge-methods)
* [LoRA Extraction](#lora-extraction)
* [Mixture of Experts Merging](#mixture-of-experts-merging)
* [Evolutionary Merge Methods](#evolutionary-merge-methods)
* [Multi-Stage Merging (`mergekit-multi`)](#multi-stage-merging-mergekit-multi)
* [Raw PyTorch Model Merging (`mergekit-pytorch`)](#raw-pytorch-model-merging-mergekit-pytorch)
* [Tokenizer Transplantation (`mergekit-tokensurgeon`)](#tokenizer-transplantation-mergekit-tokensurgeon)
* [Citation](#citation)

---

## Why Merge Models?

모델 병합은 **앙상블(ensembling)** 이나 **추가 학습 없이** 서로 다른 모델의 강점을 결합할 수 있는 강력한 기법입니다. 가중치 공간(weight space)에서 직접 연산하기 때문에 다음과 같은 장점이 있습니다:

* 여러 특화 모델을 하나의 범용 모델로 결합
* 학습 데이터 없이도 모델 간 능력 전이
* 서로 다른 모델 특성 간의 최적 트레이드오프 탐색
* 추론 비용을 유지하면서 성능 개선
* 창의적인 모델 조합을 통한 새로운 능력 생성

전통적인 앙상블 방식은 여러 모델을 동시에 실행해야 하지만, **병합된 모델은 단일 모델과 동일한 추론 비용**으로도 종종 더 나은 성능을 보입니다.

---

## Features

`mergekit`의 주요 기능:

* Llama, Mistral, GPT-NeoX, StableLM 등 다양한 아키텍처 지원
* 다양한 [merge methods](#merge-methods)
* GPU / CPU 실행 지원
* 메모리 사용량을 줄이기 위한 lazy tensor loading
* 파라미터 값에 대한 interpolated gradients (Gryphe의 BlockMerge_Gradient 아이디어 기반)
* 레이어 단위 조합("Frankenmerging")
* [Mixture of Experts merging](#mixture-of-experts-merging)
* [LoRA extraction](#lora-extraction)
* [Evolutionary merge methods](#evolutionary-merge-methods)
* 복잡한 워크플로우를 위한 [Multi-stage merging (`mergekit-multi`)](#multi-stage-merging-mergekit-multi)
* [Raw PyTorch model merging (`mergekit-pytorch`)](#raw-pytorch-model-merging-mergekit-pytorch)

---

## Installation

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .  # 패키지 설치 및 스크립트 사용 가능
```

만약 다음과 같은 에러가 발생한다면:

```text
ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode:
(A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
```

pip 버전을 21.3 이상으로 업데이트해야 합니다:

```sh
python3 -m pip install --upgrade pip
```

---

## Community & Support

* **Issues**: GitHub Issues
* **Discussions**: Arcee Discord

### Contributing

`mergekit`에 대한 기여를 환영합니다. 새로운 병합 방법, 기능 개선 아이디어가 있다면 `CONTRIBUTING.md`를 참고해 주세요.

### Community Tools

* **FrankensteinAI**: 로컬 환경이나 하드웨어 없이도 `mergekit`을 사용할 수 있는 웹 기반 플랫폼

---

## Usage

`mergekit-yaml`은 `mergekit`의 메인 엔트리 포인트입니다.

```sh
mergekit-yaml path/to/your/config.yml ./output-model-directory [--cuda] [--lazy-unpickle] [--allow-crimes]
```

병합이 완료되면 결과 모델이 `./output-model-directory`에 저장됩니다.

자세한 옵션은 다음을 참고하세요:

```sh
mergekit-yaml --help
```

### Uploading to Huggingface (영문 유지)

Hugging Face Hub 업로드 방법은 공식 문서를 참고하세요.

---

## Merge Configuration

병합 설정은 YAML 파일로 작성되며, 병합 과정을 정의합니다.

주요 필드:

* `merge_method`: 병합 방법
* `slices` / `models`: 입력 모델 정의 (서로 배타적)
* `base_model`: 일부 병합 방법에서 기준 모델
* `parameters`: 가중치 및 밀도 파라미터
* `dtype`: 연산 데이터 타입
* `tokenizer` / `tokenizer_source`: 토크나이저 구성
* `chat_template`: 채팅 템플릿 설정

---

## Parameter Specification

파라미터는 여러 수준에서 지정 가능하며, **우선순위**는 다음과 같습니다:

1. `slices.*.sources.parameters`
2. `slices.*.parameters`
3. `models.*.parameters` / `input_model_parameters`
4. `parameters`

파라미터 형태:

* **Scalar**: 단일 실수 값
* **Gradient**: 실수 리스트 (보간된 gradient)

---

## Tokenizer Configuration

토크나이저는 **권장 방식(`tokenizer`)** 또는 **레거시(`tokenizer_source`)** 방식으로 설정할 수 있습니다.

### Modern Configuration (`tokenizer`)

```yaml
tokenizer:
  source: union
  tokens:
    <token_name>:
      source: ...
      force: false
```

* `union`: 모든 입력 모델 vocab 합집합
* `base`: base model vocab 사용
* `path/to/model`: 특정 모델 vocab 사용

---

## Chat Template Configuration

```yaml
chat_template: "auto"
```

* `auto`: 가장 흔한 템플릿 자동 선택
* Built-in templates: `alpaca`, `chatml`, `llama3`, `mistral`, `exaone`
* 사용자 정의 Jinja2 템플릿 가능

---

## Merge Methods

`mergekit`은 다양한 병합 방법을 제공합니다. 각 방법의 상세 설명은 **docs/merge_methods.md**를 참고하세요.

(방법 표는 원문과 동일하므로 생략)

---

## LoRA Extraction

```sh
mergekit-extract-lora --model finetuned_model --base-model base_model --out-path output_path
```

---

## Mixture of Experts Merging

`mergekit-moe` 스크립트를 사용해 여러 dense 모델을 MoE로 병합할 수 있습니다.

---

## Multi-Stage Merging (`mergekit-multi`)

여러 단계의 병합을 하나의 YAML로 정의할 수 있습니다.

---

## Raw PyTorch Model Merging (`mergekit-pytorch`)

Transformers가 아닌 PyTorch 모델도 병합 가능합니다.

```sh
mergekit-pytorch config.yml output_dir
```

---

## Tokenizer Transplantation (`mergekit-tokensurgeon`)

토크나이저를 다른 모델로 이식하기 위한 전용 도구입니다.

---

## Citation

```bibtex
@inproceedings{goddard-etal-2024-arcees,
  title = "Arcee's MergeKit: A Toolkit for Merging Large Language Models",
  year = "2024"
}
```
