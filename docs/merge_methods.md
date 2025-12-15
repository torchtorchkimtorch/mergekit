# Merge Method Guide

## Table of Contents

- [Overview](#overview)
- [Basic Merging Methods](#basic-merging-methods)
  - [Linear (`linear`)](#linear-linear)
- [Spherical Interpolation Methods](#spherical-interpolation-methods)
  - [SLERP (`slerp`)](#slerp-slerp)
  - [NuSLERP (`nuslerp`)](#nuslerp-nuslerp)
  - [Multi-SLERP (`multislerp`)](#multi-slerp-multislerp)
  - [Karcher Mean (`karcher`)](#karcher-mean-karcher)
- [Task Vector Methods](#task-vector-methods)
  - [Task Arithmetic (`task_arithmetic`)](#task-arithmetic-task_arithmetic)
  - [TIES-Merging (`ties`)](#ties-merging-ties)
  - [DARE (`dare_linear`, `dare_ties`)](#dare-dare_linear-dare_ties)
  - [DELLA (`della`, `della_linear`)](#della-della-della_linear)
  - [Model Breadcrumbs (`breadcrumbs`, `breadcrumbs_ties`)](#model-breadcrumbs-breadcrumbs-breadcrumbs_ties)
  - [SCE (`sce`)](#sce-sce)
- [Specialized Methods](#specialized-methods)
  - [Model Stock (`model_stock`)](#model-stock-model_stock)
  - [Nearswap (`nearswap`)](#nearswap-nearswap)
  - [Arcee Fusion (`arcee_fusion`)](#arcee-fusion-arcee_fusion)
  - [Passthrough (`passthrough`)](#passthrough-passthrough)
- [Summary](#summary)
- [Contributing](#contributing)

---

## Overview

이 가이드는 `mergekit`에서 제공하는 다양한 **모델 병합(model merging) 알고리즘**에 대해 상세히 설명합니다.  
각 방법은 서로 다른 **사용 사례(use case)**, **파라미터**, 그리고 **모델 결합 방식**을 가지며, 목적에 따라 적합한 기법이 달라집니다.

---

## Basic Merging Methods

### Linear (`linear`)

**개념 (Concept):**  
입력 모델들의 파라미터를 **가중 평균(weighted average)** 으로 단순히 결합합니다. 가장 기본적이며 널리 사용되는 병합 기법입니다.

**사용 사례 (Use Cases):**

- 동일한 fine-tuning run에서 나온 여러 checkpoint를 평균내는 경우 (“model soups”)
- 아키텍처와 학습 데이터가 매우 유사한 모델들을 결합하는 경우
- 단일 모델 안에서 간단한 ensemble 효과를 얻고 싶은 경우

**입력 (Inputs):**  
2개 이상의 모델을 입력으로 받습니다. 일반적으로 `base_model`은 사용하지 않습니다.

**주요 파라미터 (Key Parameters):**

- `weight` (모델별): 각 모델이 평균에 기여하는 비중
- `normalize` (전역): `true`일 경우(기본값), weight의 합이 1이 되도록 정규화

**참고 문헌 (Reference):**  
[Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time](https://arxiv.org/abs/2203.05482)

---

## Spherical Interpolation Methods

### SLERP (`slerp`)

**개념 (Concept):**  
두 모델 사이의 weight 공간에서 **Spherical Linear Interpolation (SLERP)** 을 수행합니다.  
이는 hypersphere 상의 경로를 따라 보간을 수행하여, 결과 모델이 원래 모델들과 유사한 **norm(크기)** 을 유지하도록 합니다.

**사용 사례 (Use Cases):**

- 두 개의 서로 다른 모델 사이에서 부드러운 전이(intermediate model)를 생성
- 서로 다른 특성을 가진 두 모델 사이의 공간을 탐색

**입력 (Inputs):**  
정확히 2개의 모델이 필요하며, 그중 하나는 반드시 `base_model`로 지정되어야 합니다.

**주요 파라미터 (Key Parameters):**

- `t` (전역): 보간 계수  
  - `t=0` → `base_model`  
  - `t=1` → 다른 모델

**참고 문헌 (Reference):**  
[Wikipedia: Slerp](https://en.wikipedia.org/wiki/Slerp)

---

### NuSLERP (`nuslerp`)

**개념 (Concept):**  
SLERP를 확장한 방식으로, 더 유연한 설정과 빠른 실행을 제공합니다.  
`base_model`이 없는 경우에는 두 모델 간 SLERP를 직접 수행합니다.  
`base_model`이 주어지면, 각 모델과 base 간의 **task vector**를 계산한 뒤, 이 task vector들에 대해 SLERP를 수행하고 결과를 다시 `base_model`에 더합니다.

**사용 사례 (Use Cases):**

- SLERP와 유사하지만 두 모델 간 가중치를 더 세밀하게 조정하고 싶은 경우
- 기존 `slerp` 동작을 재현하고 싶은 경우  
  (base_model 없이, 첫 번째 모델 weight를 `1-t`, 두 번째를 `t`로 설정)
- 공통 조상(base_model)을 기준으로 **상대적인 변화(task vector)** 를 보간하고 싶은 경우

**입력 (Inputs):**  
정확히 2개의 모델이 필요합니다.  
`base_model`은 선택 사항이며, 두 모델과는 다른 모델이어야 합니다.

**주요 파라미터 (Key Parameters):**

- `weight` (모델별): 두 모델의 상대적 가중치  
  - 보간 계수는  
    `t = model2_weight / (model1_weight + model2_weight)` 로 계산됨
- `nuslerp_flatten` (전역): `false`일 경우 row/column 단위로 SLERP 수행  
  (기본값: `true`)
- `nuslerp_row_wise` (전역):  
  `nuslerp_flatten=false`일 때 `true`이면 column이 아닌 **row vector** 단위로 SLERP 수행  
  (기본값: `false`)

---

### Multi-SLERP (`multislerp`)

**개념 (Concept):**  
2개 초과의 모델에 대해 hypersphere 상에서 **barycentric interpolation**을 수행합니다.  
가중 Euclidean mean에서 tangent space로 투영한 뒤 보간을 수행하고, 다시 원래 공간으로 되돌립니다.

**사용 사례 (Use Cases):**

- 여러 모델의 spherical average를 구하고 싶은 경우
- 관련된 여러 모델의 weight 공간에서 중심점(center)을 찾고 싶은 경우

**입력 (Inputs):**  
2개 이상의 모델을 입력으로 받습니다.  
선택적으로 `base_model`을 지정하여 task vector 공간에서 동작하게 할 수 있습니다.

**주요 파라미터 (Key Parameters):**

- `weight` (모델별): 각 모델의 상대적 가중치
- `normalize_weights` (전역): `true`일 경우(기본값), weight 정규화
- `eps` (전역): 수치 안정성을 위한 작은 상수 (기본값: `1e-8`)

---

### Karcher Mean (`karcher`)

**개념 (Concept):**  
입력 모델 파라미터들의 **Karcher mean**  
(= Riemannian barycenter, Fréchet mean)을 계산합니다.  
이는 manifold 상의 점들을 평균내는 기하학적으로 정당한 방법입니다.

**사용 사례 (Use Cases):**

- 서로 다른 여러 모델들 사이에서 “중앙” 또는 “평균” 모델을 찾고 싶은 경우
- weight 공간에서 멀리 떨어진 모델들을 단순 평균보다 더 안정적으로 결합하고 싶은 경우

**입력 (Inputs):**  
2개 이상의 모델을 입력으로 받으며, `base_model`은 사용하지 않습니다.

**주요 파라미터 (Key Parameters):**

- `max_iter` (전역): 알고리즘 최대 반복 횟수 (기본값: `10`)
- `tol` (전역): 수렴 허용 오차 (기본값: `1e-5`)

**참고 문헌 (Reference):**  
[Wikipedia: Karcher mean](https://en.wikipedia.org/wiki/Karcher_mean)

---

## Task Vector Methods

*아래 방법들은 모두 fine-tuned 모델과 `base_model`의 차이를 나타내는  
**task vector** 개념을 기반으로 합니다.*

### Task Arithmetic (`task_arithmetic`)

**개념 (Concept):**  
각 모델에서 `base_model`을 빼서 task vector를 계산합니다.  
이 task vector들을 가중 평균한 뒤, 다시 `base_model`에 더합니다.

**사용 사례 (Use Cases):**

- 공통 조상(base)에서 파생된 여러 모델의 능력을 결합
- 특정 능력(코딩, instruction-following 등)을 다른 모델에 이전
- 스타일이나 행동을 작은 task vector로 조정

**입력 (Inputs):**  
`base_model` 1개 + 1개 이상의 추가 모델 필요

**주요 파라미터 (Key Parameters):**

- `weight` (모델별): 각 task vector의 가중치
- `lambda` (전역): 합쳐진 task vector에 곱해지는 스케일 계수 (기본값: `1.0`)

**참고 문헌 (Reference):**  
[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)

---

### TIES-Merging (`ties`)

**개념 (Concept):**  
Task Arithmetic을 확장한 방식으로,  
task vector를 **sparsify**하고 **sign consensus 알고리즘**을 적용하여  
모델 간 간섭(interference)을 줄입니다.

**사용 사례 (Use Cases):**

- 많은 수의 모델을 효과적으로 병합
- 파라미터 간 충돌 및 negative synergy 감소

**입력 (Inputs):**  
2개 이상의 모델 + 1개의 `base_model`

**주요 파라미터 (Key Parameters):**

- `weight` (모델별)
- `density` (모델별): 유지할 파라미터 비율
- `lambda` (전역): Task Arithmetic과 동일

**참고 문헌 (Reference):**  
[TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)

---

### DARE (`dare_linear`, `dare_ties`)

**개념 (Concept):**  
TIES와 유사하지만, 무작위 pruning과 **rescaling**을 사용하여  
원본 모델 성능을 더 잘 보존하도록 설계된 방법입니다.

**변형 (Variants):**

- `dare_linear`: TIES sign consensus 없이 DARE pruning
- `dare_ties`: TIES sign consensus 포함

**사용 사례 (Use Cases):**

- 여러 fine-tuned 모델을 안정적으로 결합
- 일부 시나리오에서 TIES보다 더 좋은 성능

**입력 (Inputs):**  
2개 이상의 모델 + 1개의 `base_model`

**주요 파라미터 (Key Parameters):**

- `weight` (모델별)
- `density` (모델별)
- `lambda` (전역)
- `rescale` (전역, `dare_linear`): 기본값 `true`

**참고 문헌 (Reference):**  
[Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)

---

### DELLA (`della`, `della_linear`)

**개념 (Concept):**  
DARE를 확장하여, task vector의 각 row 내에서 **파라미터 크기(magnitude)** 에 기반한  
adaptive pruning을 수행합니다.  
큰 magnitude를 가진 파라미터일수록 유지 확률이 높고,  
작은 magnitude일수록 낮습니다. 이후 DARE와 유사한 rescaling을 적용합니다.

**변형 (Variants):**

- `della`: TIES sign consensus 포함
- `della_linear`: TIES 없이 DELLA pruning

**사용 사례 (Use Cases):**

- 중요한 변화(large magnitude)를 우선적으로 보존하고 싶은 경우
- 더 세밀한 pruning 제어가 필요한 병합

**입력 (Inputs):**  
2개 이상의 모델 + 1개의 `base_model`

**주요 파라미터 (Key Parameters):**

- `weight` (모델별)
- `density` (모델별)
- `epsilon` (모델별): keep probability 범위 조절  
  (`density - epsilon` ~ `density + epsilon`)
- `lambda` (전역)

**참고 문헌 (Reference):**  
[DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling](https://arxiv.org/abs/2406.11617)

---

### Model Breadcrumbs (`breadcrumbs`, `breadcrumbs_ties`)

**개념 (Concept):**  
task vector에서 **가장 작은 값과 가장 큰 값(이상치)** 을 모두 제거하여  
중간 영역(mid-range)의 변화만을 남기는 방식입니다.

1. 가장 큰 magnitude 상위 `gamma` 비율 제거
2. 목표 `density`를 만족하도록 가장 작은 magnitude 제거

**변형 (Variants):**

- `breadcrumbs`: TIES 없이
- `breadcrumbs_ties`: TIES 포함

**입력 (Inputs):**  
2개 이상의 모델 + 1개의 `base_model`

**주요 파라미터 (Key Parameters):**

- `weight` (모델별)
- `gamma` (모델별): 가장 큰 magnitude 제거 비율
- `density` (모델별): 최종 유지 비율
- `lambda` (전역)

**참고 문헌 (Reference):**  
[Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks](https://arxiv.org/abs/2312.06795)

---

### SCE (`sce`)

**개념 (Concept):**  
SCE(Select, Calculate, Erase)는 **matrix-level 적응형 병합** 방법입니다.

1. **Select:** variance 기반 마스킹
2. **Calculate:** matrix 단위 가중치 계산
3. **Erase:** TIES sign consensus 적용

이후 정규화된 task vector를 `base_model`에 더합니다.

**입력 (Inputs):**  
2개 이상의 모델 + 1개의 `base_model`

**주요 파라미터 (Key Parameters):**

- `select_topk` (전역): variance 기준으로 유지할 파라미터 비율 (기본값: `1.0`)

**참고 문헌 (Reference):**  
[FuseChat: Knowledge Fusion of Chat Models](https://arxiv.org/abs/2408.07990)

---

## Specialized Methods

### Model Stock (`model_stock`)

**개념 (Concept):**  
`base_model` 대비 다른 모델들의 task vector cosine similarity를 이용해  
최적의 interpolation weight를 계산한 뒤 선형 보간을 수행합니다.

**입력 (Inputs):**  
최소 3개 모델 필요 (`base_model` + 2개 이상)

**주요 파라미터 (Key Parameters):**

- `filter_wise` (전역): row 단위 계산 여부 (기본값 `false`)

**참고 문헌 (Reference):**  
[Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/abs/2403.19522)

---

### Nearswap (`nearswap`)

**개념 (Concept):**  
두 모델의 파라미터 차이가 **작은 부분만 강하게 보간**하는 방식입니다.

**입력 (Inputs):**  
정확히 2개 모델 (`base_model` 필수)

**주요 파라미터 (Key Parameters):**

- `t` (전역): 보간 강도

**알고리즘 (Algorithm):**

weight = (t / |base - secondary|).clamp(0, 1)
output = weight * secondary + (1 - weight) * base


**참고 문헌 (Reference):**  
[QuartetAnemoi-70B-t0.0001 on Hugging Face](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001)

---

### Arcee Fusion (`arcee_fusion`)

**개념 (Concept):**  
파라미터 차이와 KL divergence 기반 중요도를 계산하여  
동적으로 fusion mask를 생성하는 병합 방식입니다.

**입력 (Inputs):**  
정확히 2개 모델 (`base_model` 필수)

**참고 문헌 (Reference):**  
[MergeKit v0.1 Release Blog](https://www.arcee.ai/blog/meet-mergekit-v0-1-arcee-fusion-expanded-model-support-multi-gpu-acceleration)

---

### Passthrough (`passthrough`)

**개념 (Concept):**  
아무런 변경 없이 입력 모델의 텐서를 그대로 통과시키는 no-op 방식입니다.

**사용 사례 (Use Cases):**

- 특정 layer만 선택적으로 조합하는 Frankenmerge
- 복잡한 `slices` 설정의 빌딩 블록

**입력 (Inputs):**  
정확히 1개 모델

**주요 파라미터 (Key Parameters):**

- `scale` (선택, 모델별): 텐서 스케일 조정

---

## Summary

Merge method는 각각 서로 다른 목적과 설계 철학을 가집니다.  
모델 수, 관계, 원하는 최종 특성에 따라 적절한 방법이 달라집니다.

초보자라면 `linear`, `nuslerp`, `task_arithmetic`부터 시작하는 것이 좋으며,  
보다 고급 시나리오에서는 `ties`, `dare_ties`, `della` 같은 방법이  
모델 간 간섭을 줄이면서 장점을 유지하는 데 효과적입니다.

“최고의” merge 방법은 존재하지 않으며,  
대부분의 경우 **실험과 튜닝이 핵심**입니다.  
다양한 설정을 적극적으로 시도해 보시길 권장합니다. Happy merging!

---

## Contributing

새로운 merge 방법 아이디어나 기존 방법 개선 제안은 언제든 환영합니다.  
자세한 내용은 다음 문서를 참고하세요:

- [Contributing Guide](../CONTRIBUTING.md)
- [Creating a Merge Method](create_a_merge_method.md)



# Merge Method Guide

## Table of Contents

- [Overview](#overview)
- [Basic Merging Methods](#basic-merging-methods)
  - [Linear (`linear`)](#linear-linear)
- [Spherical Interpolation Methods](#spherical-interpolation-methods)
  - [SLERP (`slerp`)](#slerp-slerp)
  - [NuSLERP (`nuslerp`)](#nuslerp-nuslerp)
  - [Multi-SLERP (`multislerp`)](#multi-slerp-multislerp)
  - [Karcher Mean (`karcher`)](#karcher-mean-karcher)
- [Task Vector Methods](#task-vector-methods)
  - [Task Arithmetic (`task_arithmetic`)](#task-arithmetic-task_arithmetic)
  - [TIES-Merging (`ties`)](#ties-merging-ties)
  - [DARE (`dare_linear`, `dare_ties`)](#dare-dare_linear-dare_ties)
  - [DELLA (`della`, `della_linear`)](#della-della-della_linear)
  - [Model Breadcrumbs (`breadcrumbs`, `breadcrumbs_ties`)](#model-breadcrumbs-breadcrumbs-breadcrumbs_ties)
  - [SCE (`sce`)](#sce-sce)
- [Specialized Methods](#specialized-methods)
  - [Model Stock (`model_stock`)](#model-stock-model_stock)
  - [Nearswap (`nearswap`)](#nearswap-nearswap)
  - [Arcee Fusion (`arcee_fusion`)](#arcee-fusion-arcee_fusion)
  - [Passthrough (`passthrough`)](#passthrough-passthrough)
- [Summary](#summary)
- [Contributing](#contributing)

## Overview

This guide provides detailed information about the various model merging algorithms available in `mergekit`. Each method has specific use cases, parameters, and applications for combining machine learning models.

## Basic Merging Methods

### Linear (`linear`)

**Concept:** Computes a simple weighted average of the parameters from the input models. This is one of the most basic and widely used merging techniques.

**Use Cases:**

- Averaging multiple checkpoints of the same fine-tuning run ("model soups")
- Combining models with very similar architectures and training data
- Simple ensemble-like behavior in a single model

**Inputs:** Takes 2 or more models. No `base_model` is typically used.

**Key Parameters:**

- `weight` (per-model): The contribution of each model to the average
- `normalize` (global): If `true` (default), weights are normalized to sum to 1

**Reference:** [Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time](https://arxiv.org/abs/2203.05482)

---

## Spherical Interpolation Methods

### SLERP (`slerp`)

**Concept:** Performs Spherical Linear Interpolation in the weight space between two models. This creates a path along a hypersphere, ensuring the interpolated model maintains a similar "norm" or "magnitude" to the original models.

**Use Cases:**

- Creating smooth transitions or intermediate points between two distinct models
- Exploring the space between two models with potentially different capabilities

**Inputs:** Requires exactly 2 models. One model must be specified as `base_model`.

**Key Parameters:**

- `t` (global): Interpolation factor. `t=0` yields the `base_model`, `t=1` yields the other model

**Reference:** [Wikipedia: Slerp](https://en.wikipedia.org/wiki/Slerp)

### NuSLERP (`nuslerp`)

**Concept:** An enhanced version of SLERP offering more flexible configuration and faster execution. It allows SLERP between two models directly. If a `base_model` is provided, NuSLERP calculates task vectors (the difference between each of the two main models and the base model) and then performs SLERP on these task vectors before adding the result back to the `base_model`.

**Use Cases:**

- Similar to SLERP, but with more control over weighting for the two primary models.
- To replicate the behavior of the original slerp method (if `base_model` is *not* used, and weights are set to `1-t` for the first model and `t` for the second).
- To perform SLERP on task vectors when a `base_model` is provided, allowing for interpolation of model changes *relative* to a common ancestor.

**Inputs:** Requires exactly 2 models. A `base_model` can optionally be provided (it must be distinct from the two main models).

**Key Parameters:**

- `weight` (per-model): Relative weighting for each of the two main models. These are used to calculate the interpolation factor `t` (where `t = model2_weight / (model1_weight + model2_weight)`).
- `nuslerp_flatten` (global): If `false`, performs row/column-wise interpolation. Default `true`
- `nuslerp_row_wise` (global): If `true` (and `nuslerp_flatten` is `false`), SLERPs row vectors instead of column vectors. Default `false`

### Multi-SLERP (`multislerp`)

**Concept:** Implements barycentric interpolation on a hypersphere for more than two models. It projects points onto a tangent space at their weighted Euclidean mean, performs interpolation, and projects back.

**Use Cases:**

- Creating a spherical average of multiple models
- Finding a central point in the weight space of several related models

**Inputs:** Takes 2 or more models. A `base_model` can optionally be provided to operate in task vector space.

**Key Parameters:**

- `weight` (per-model): Relative weighting for each model
- `normalize_weights` (global): If `true` (default), weights are normalized
- `eps` (global): Small constant for numerical stability. Default `1e-8`

### Karcher Mean (`karcher`)

**Concept:** Computes the Karcher mean (also known as the Riemannian barycenter or Fréchet mean) of the input model parameters. This provides a geometrically sound way to average points on a manifold, which is suitable for model weights.

**Use Cases:**

- Finding a "central" or "average" model among a set of diverse models in a way that respects the geometry of the weight space
- More robust averaging than simple linear averaging, especially for models far apart in weight space

**Inputs:** Takes 2 or more models. No `base_model` is used.

**Key Parameters:**

- `max_iter` (global): Maximum iterations for the Karcher mean algorithm. Default `10`
- `tol` (global): Convergence tolerance. Default `1e-5`

**Reference:** [Wikipedia: Karcher mean](https://en.wikipedia.org/wiki/Karcher_mean)

---

## Task Vector Methods

*The following methods build upon the concept of "task vectors," which represent the difference between a fine-tuned model and a base model.*

### Task Arithmetic (`task_arithmetic`)

**Concept:** Computes "task vectors" for each model by subtracting a `base_model`. These task vectors are then combined as a weighted average and added back to the `base_model`.

**Use Cases:**

- Combining skills from multiple models fine-tuned from a common ancestor
- Transferring specific capabilities (e.g., coding ability, instruction following) from one model to another
- Steering style or behavior of a model by adding small task vectors from other models

**Inputs:** Requires a `base_model` and one or more other models.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector in the merge
- `lambda` (global): Scaling factor applied to the summed task vectors before adding back to the base. Default `1.0`

**Reference:** [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)

### TIES-Merging (`ties`)

**Concept:** Builds on Task Arithmetic by sparsifying task vectors and applying a sign consensus algorithm. This helps to resolve interference when merging multiple models and retain more of their individual strengths.

**Use Cases:**

- Merging a larger number of models effectively
- Reducing parameter interference and negative synergy between merged models

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector
- `density` (per-model): Fraction of weights to retain in each sparsified task vector
- `lambda` (global): As in Task Arithmetic

**Reference:** [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)

### DARE (`dare_linear`, `dare_ties`)

**Concept:** Similar to TIES, DARE sparsifies task vectors to reduce interference. However, DARE uses random pruning with a novel rescaling technique to better match the performance of the original models.

**Variants:**

- `dare_linear`: DARE pruning without the TIES sign consensus
- `dare_ties`: DARE pruning *with* the TIES sign consensus

**Use Cases:**

- Robustly combining multiple fine-tuned models, often yielding better performance than TIES in some scenarios

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector
- `density` (per-model): Fraction of weights to retain after random pruning
- `lambda` (global): As in Task Arithmetic
- `rescale` (global, for `dare_linear`): If `true` (default), applies DARE's rescaling

**Reference:** [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)

### DELLA (`della`, `della_linear`)

**Concept:** Extends DARE by using adaptive pruning based on parameter magnitudes within each row of the delta parameters (task vectors). It calculates keep probabilities for each parameter: parameters with larger magnitudes within a row are assigned higher probabilities of being kept, while parameters with smaller magnitudes are assigned lower probabilities. These keep probabilities are scaled to range from `density - epsilon` (for the smallest magnitude element in a row) to `density + epsilon` (for the largest magnitude element in a row). This method aims to retain important changes while reducing interference, followed by DARE-like rescaling.

**Variants:**

- `della`: DELLA pruning with TIES sign consensus
- `della_linear`: DELLA pruning without TIES sign consensus

**Use Cases:**

- Fine-grained control over pruning by prioritizing parameters with larger magnitude changes
- Combining models where preserving the most significant changes is crucial

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector
- `density` (per-model): Target fraction of weights to retain in differences from the base model
- `epsilon` (per-model): Defines the half-width of the range for keep probabilities. Keep probabilities for parameters in a row will range from `density - epsilon` to `density + epsilon`, mapped from the smallest to largest magnitude parameters in that row, respectively. `epsilon` must be chosen such that `density - epsilon > 0` and `density + epsilon < 1`.
- `lambda` (global): As in Task Arithmetic

**Reference:** [DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling](https://arxiv.org/abs/2406.11617)

### Model Breadcrumbs (`breadcrumbs`, `breadcrumbs_ties`)

**Concept:** An extension of task arithmetic designed to sparsify task vectors by pruning parameters with both the smallest and the largest absolute magnitudes (often considered outliers). This method operates in two main steps on the task vector (the difference between a fine-tuned model and the `base_model`):

1. First, a `gamma` fraction of the parameters with the *largest* absolute magnitudes are identified for removal.
2. Then, parameters with the *smallest* absolute magnitudes are identified for removal. The quantity of these smallest parameters to remove is determined such that the final `density` of parameters *retained* in the task vector is achieved, after accounting for the largest ones removed.

The intention is to isolate and merge the "meaty," mid-range magnitude changes from the task vector, potentially filtering out noise (smallest changes) and overly dominant or conflicting large changes (largest changes).

**Variants:**

- `breadcrumbs`: Model Breadcrumbs pruning without TIES sign consensus
- `breadcrumbs_ties`: Model Breadcrumbs pruning *with* TIES sign consensus

**Use Cases:**

- Merging models where extreme parameter changes might be detrimental or noisy
- Refining task vectors by focusing on mid-range modifications, removing both the least significant and most extreme changes

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector.
- `gamma` (per-model): The fraction of parameters with the *largest* absolute magnitudes in the task vector to be pruned (removed). For example, a `gamma` of `0.01` targets the removal of the top 1% of parameters with the highest absolute values. This parameter corresponds to `β` (beta) as described in the reference paper.
- `density` (per-model): The final target fraction of parameters to *retain* in the task vector after both pruning steps (removal of largest `gamma` fraction and a corresponding fraction of smallest magnitude parameters).
  - The fraction of parameters with the *smallest* absolute magnitudes that will be pruned is calculated based on `density` and `gamma`. Specifically, it is `max(0, 1.0 - density - gamma)`.
  - **Example:** If `density: 0.9` and `gamma: 0.01`:
    - The top `0.01` (1%) largest magnitude parameters are removed.
    - The bottom `1.0 - 0.9 - 0.01 = 0.09` (9%) smallest magnitude parameters are also removed.
    - This results in `0.9` (90%) of the parameters being retained.
  - **Edge Case:** If `gamma` is set high enough such that `gamma >= 1.0 - density` (meaning `1.0 - density - gamma <= 0`), then the number of largest magnitude parameters actually pruned will be adjusted to `1.0 - density`, and no smallest magnitude parameters will be pruned (i.e., the fraction of smallest parameters pruned becomes 0). This ensures the `density` target is always respected and represents the fraction of parameters kept.
- `lambda` (global): As in Task Arithmetic.

**Reference:** [Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks](https://arxiv.org/abs/2312.06795)

### SCE (`sce`)

**Concept:** The SCE (Select, Calculate, Erase) method performs adaptive matrix-level merging. It first computes task vectors (differences from the `base_model`). Then, it follows a three-step process for each parameter matrix (tensor):

1. **Select (Variance-Based Masking):** Optionally, parameter *positions* that show low variance across the different models' task vectors are identified and zeroed out. This is controlled by the `select_topk` parameter. If `select_topk < 1.0`, only the top `select_topk` fraction of parameter positions with the highest variance are kept active in the task vectors for subsequent steps.
2. **Calculate (Weighting):** Matrix-level merging coefficients (weights) are calculated for each model's task vector. These weights are derived from the mean of the squares of the elements within each task vector and are normalized across the models.
3. **Erase (Sign Consensus):** The sign-consensus algorithm from TIES is applied to the task vectors.

Finally, the (variance-selected, calculated-weighted, and sign-agreed) task vectors are summed together, normalized by the sum of the effective applied weights at each position, and then added back to the `base_model`.

**Use Cases:**

- Dynamically weighting the contribution of different models at the matrix level based on parameter variance and calculated importance
- Useful when some models contribute more significantly or consistently to certain parameter matrices than others
- Merging models by focusing on high-variance, consistently signed changes

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `select_topk` (global): The fraction of parameter positions to retain based on their variance values across the different input models' task vectors. For each parameter position, variance is calculated across all task vectors. Only positions corresponding to the `select_topk` fraction with the highest variances are kept (i.e., their values in all task vectors are preserved for the next steps). Positions with lower variance are zeroed out in all task vectors. Set to 1.0 (default) to disable this variance-based selection step. This corresponds to `τ` (tau) in the reference paper.

**Reference:** [FuseChat: Knowledge Fusion of Chat Models](https://arxiv.org/abs/2408.07990)

---

## Specialized Methods

### Model Stock (`model_stock`)

**Concept:** Uses geometric properties of fine-tuned models relative to a base_model to compute an optimized interpolation weight. It then performs a linear interpolation between the `base_model` and the average of the other input models using this computed weight. Specifically, task vectors (differences between other models and the `base_model`) are used to calculate pairwise cosine similarities. The average of these similarities informs the interpolation factor `t`. The final merged tensor is `t * average_of_other_models + (1 - t) * base_model`.

**Use Cases:**

- Finding effective weights for linearly combining multiple models where one model serves as a clear reference (`base_model`).
- When a more principled, data-driven approach to linear interpolation between a base and a group of variants is desired over manual weight tuning.
- Particularly strong for combining different training runs that were fine-tuned from the same `base_model` over the same or similar datasets.

**Inputs:** Requires at least 3 models: one `base_model` and at least two other models.

**Key Parameters:**

- `filter_wise` (global): If `true`, weight calculation is per-row rather than per-tensor (not generally recommended). Default `false`

**Reference:** [Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/abs/2403.19522)

### Nearswap (`nearswap`)

**Concept:** Interpolates the base model with parameters from a secondary model primarily where they are already similar. The interpolation strength towards the secondary model is inversely proportional to the absolute difference of their parameters, modulated by the `t` parameter. When the parameters are similar, the interpolation is stronger, and when they are different, it is weaker.

**Use Cases:**

- Selectively pulling in similar parameters from a secondary model while preserving different parameters from the base model
- Fine-grained parameter-wise merging that respects the existing structure of the base model

**Inputs:** Requires exactly 2 models. One model must be specified as `base_model`.

**Key Parameters:**

- `t` (global): Controls the interpolation strength. Higher values increase the influence of the secondary model for similar parameters

**Algorithm:** For each parameter, computes `weight = (t / |base - secondary|).clamp(0, 1)`, then returns `weight * secondary + (1 - weight) * base`

**Reference:** [QuartetAnemoi-70B-t0.0001 on Hugging Face](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001)

### Arcee Fusion (`arcee_fusion`)

**Concept:** Merges two models by dynamically identifying and fusing important parameter changes. It calculates importance scores based on parameter differences and KL divergence, then uses a dynamic threshold to create a fusion mask.

**Use Cases:**

- Intelligently combining two models by prioritizing the most salient differences

**Inputs:** Requires exactly 2 models. One model must be specified as `base_model`.

**Key Parameters:** None beyond standard model selection

**Reference:** [MergeKit v0.1 Release Blog](https://www.arcee.ai/blog/meet-mergekit-v0-1-arcee-fusion-expanded-model-support-multi-gpu-acceleration)

### Passthrough (`passthrough`)

**Concept:** A no-op merge method that simply passes input tensors through unmodified from a single input model.

**Use Cases:**

- Layer-stacking or "Frankenmerging" where you assemble a model from specific unmodified layer ranges or individual tensors from one or more "donor" models
- Useful as a building block in more complex `slices` configurations

**Inputs:** Takes exactly 1 model.

**Key Parameters:**

- `scale` (per-model, optional): A scalar to multiply the tensor by. Useful for scaling specific layers, e.g., `{"filter": "down_proj", "value": 0.5}`

---

## Summary

Merge methods serve different purposes and have different design goals. The choice of method depends on your specific use case, including the number of models, their relationships, and the desired characteristics of the final merged model.

For beginners, starting with `linear`, `nuslerp`, or `task_arithmetic` can provide good results. For more advanced use cases, methods like `ties`, `dare_ties`, or `della` offer sophisticated ways to handle interference between multiple models while preserving their individual strengths. There is no "best" merge method; the right choice depends on your specific needs and the models you are working with, and selection is often more art than science. I encourage you to experiment with different methods and parameters vigorously to both find the best results for your use case and to learn more about how these methods work. Happy merging!

## Contributing

If you have ideas for new merge methods or improvements to existing ones, we'd be glad to have you involved! Check out the [Contributing Guide](../CONTRIBUTING.md) and [Creating a Merge Method](create_a_merge_method.md) for more information on how to get started.
