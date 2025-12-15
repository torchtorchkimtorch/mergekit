# mergekit-multi: Multi-Stage Model Merging

## mergekit-multi란?

`mergekit-multi`는 **여러 단계로 이루어진 복잡한 모델 머지(workflow)**를 실행하기 위한 커맨드라인 도구입니다. 서로 의존성이 있는 머지 단계를 하나의 파이프라인으로 정의하고 실행할 수 있습니다.

이를 통해 다음과 같은 작업이 가능합니다:

1. 여러 merge 작업을 체인처럼 연결
2. 이전 merge의 출력을 다음 merge의 입력으로 사용
3. merge 단계 간 의존성을 자동으로 처리
4. 중간 결과를 캐시하여 재실행 속도 향상

---

## Usage

기본 커맨드 구조:

```bash
mergekit-multi <config.yaml> \
  --intermediate-dir ./intermediates \
  ([--out-path ./final-merge] | if config has unnamed merge) \
  [options]
```

* `config.yaml`: multi-stage merge 설정 파일
* `--intermediate-dir`: 중간 merge 결과 저장 디렉터리
* `--out-path`: **이름 없는(final) merge가 있을 때만** 최종 출력 경로로 사용됨

---

## Configuration File Format

YAML 파일 하나에 여러 개의 merge 설정을 정의하며, 각 설정은 `---` 로 구분합니다.

각 merge 블록은 다음을 포함해야 합니다:

* `name`: 중간(intermediate) merge를 식별하기 위한 고유 이름

  * **최종 merge에는 `name`을 쓰지 않습니다**
* 일반적인 mergekit 설정 파라미터 (`merge_method`, `models`, `parameters` 등)

---

## Example: Final Merge 포함 (`multimerge.yaml`)

```yaml
name: first-merge
merge_method: linear
models:
  - model: mistralai/Mistral-7B-v0.1
  - model: BioMistral/BioMistral-7B
parameters:
  weight: 0.5
---
name: second-merge
merge_method: slerp
base_model: first-merge  # 이전 merge 결과를 참조
models:
  - model: NousResearch/Hermes-2-Pro-Mistral-7B
parameters:
  t: 0.5
---
# Final merge (name 없음)
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: second-merge
    parameters:
      density: 0.6
      weight: 0.5
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      density: 0.8
      weight: 0.5
```

### 핵심 포인트

* `base_model: first-merge`, `base_model: second-merge` 처럼 **이전 merge의 name을 그대로 참조 가능**
* 마지막 merge는 `name`이 없으며, 이 경우 `--out-path`가 최종 결과 경로가 됨

---

## Example: 모든 Merge에 name이 있는 경우

```yaml
name: first-merge
merge_method: task_arithmetic
...
---
name: second-merge
merge_method: slerp
...
---
name: third-merge
merge_method: linear
...
```

이 경우:

* 모든 merge는 intermediate 결과로 취급됨
* `--out-path`는 사용되지 않음
* 각 결과는 `--intermediate-dir/<name>` 경로에 저장됨

---

## Key Options

* `--intermediate-dir`

  * 중간 merge 결과를 저장할 디렉터리

* `--out-path`

  * **name이 없는 merge가 하나 있을 때만** 최종 출력 경로로 사용

* `--lazy / --no-lazy`

  * `--lazy` (기본값: true): 이미 결과가 존재하는 merge는 재실행하지 않음
  * `--no-lazy`: 모든 merge 단계를 강제로 다시 실행

* 기타 standard mergekit 옵션들 사용 가능:

  * `--cuda`
  * `--out-shard-size`
  * `--multi-gpu`
  * 등등

---

## How It Works

`mergekit-multi`를 실행하면 내부적으로 다음 과정을 거칩니다:

1. 모든 merge 설정을 분석하여 **의존성 그래프**를 구성
2. topological sort를 통해 **올바른 실행 순서**를 자동 결정
3. merge를 순차적으로 실행하면서:

   * 이전 단계 출력이 필요한 경우 자동으로 연결
4. 각 intermediate merge 결과를 `--intermediate-dir`에 저장

기본적으로는 이미 결과 파일이 존재하는 merge 단계는 **스킵**됩니다.

모든 단계를 처음부터 다시 실행하고 싶다면:

```bash
mergekit-multi config.yaml --no-lazy
```

를 사용하면 됩니다.


# mergekit-multi: Multi-Stage Model Merging

## What is mergekit-multi?

`mergekit-multi` is a command-line tool for executing complex model merging workflows with multiple interdependent stages. It allows you to:

1. Chain multiple merge operations together
2. Use outputs from previous merges as inputs to subsequent ones
3. Automatically handle dependencies between merge steps
4. Cache intermediate results for faster re-runs

## Usage

Basic command structure:
```bash
mergekit-multi <config.yaml> \
  --intermediate-dir ./intermediates \
  ([--out-path ./final-merge] | if config has unnamed merge) \
  [options]
```

## Configuration File Format

Create a YAML file with multiple merge configurations separated by `---`. Each should contain:

- `name`: Unique identifier for intermediate merges (except final merge)
- Standard mergekit configuration parameters

Example with Final Merge (`multimerge.yaml`):
```yaml
name: first-merge
merge_method: linear
models:
  - model: mistralai/Mistral-7B-v0.1
  - model: BioMistral/BioMistral-7B
parameters:
  weight: 0.5
---
name: second-merge
merge_method: slerp
base_model: first-merge  # Reference previous merge
models:
  - model: NousResearch/Hermes-2-Pro-Mistral-7B
parameters:
  t: 0.5
---
# Final merge (no name)
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: second-merge
    parameters:
      density: 0.6
      weight: 0.5
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      density: 0.8
      weight: 0.5
```

### Example with All Named Merges:
```yaml
name: first-merge
merge_method: task_arithmetic
...
---
name: second-merge
merge_method: slerp
...
---
name: third-merge
merge_method: linear
...
```

## Key Options

- `--intermediate-dir`: Directory to store partial merge results
- `--out-path`: Output path for final merge (only applies when one merge has no `name`)
- `--lazy/--no-lazy`: Don't rerun existing intermediate merges (default: true)
- Standard mergekit options apply (e.g., `--cuda`, `--out-shard-size`, `--multi-gpu`)

## How It Works

When you run `mergekit-multi`, it topologically sorts your merge configurations to determine the correct order of execution. The merges are then processed sequentially, using outputs from previous steps as inputs for subsequent ones as needed.

All intermediate merges are saved in your specified `--intermediate-dir` using their configured names. By default, the tool will skip any merge operations that already have existing output files. To force re-execution of all merges, use the `--no-lazy` flag.
