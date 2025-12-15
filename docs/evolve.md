# mergekit-evolve

`mergekit-evolve`는 진화 알고리즘(CMA-ES)을 사용하여 **모델 병합(merge)의 파라미터를 자동으로 최적화**하는 스크립트입니다. 이 도구는 SakanaAI의 논문 *[Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187)*, 특히 **parameter-space 접근법**에서 영감을 받았습니다.

`mergekit-evolve`는 EleutherAI의 [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)를 사용하여 **평가 지표(scoring function)**를 정의하고 계산합니다. 단일 노드 환경뿐만 아니라 **Ray 클러스터**에서도 실행할 수 있으며, 사용자의 컴퓨팅 환경에 따라 서로 다른 스케줄링 전략을 제공합니다.

---

## Installation

`mergekit`을 `evolve` 기능(필요 시 `vllm` 포함)과 함께 설치합니다:

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .[evolve,vllm]
```

만약 기존 PyTorch 환경이 잘 동작하고 있었는데, vLLM의 구버전 설치로 인해 **flash-attention**이 깨졌다면 아래 명령으로 복구할 수 있습니다:

```sh
pip uninstall flash-attn
pip cache purge
pip install flash-attn
```

---

## Configuration

`mergekit-evolve`는 **YAML 설정 파일**을 입력으로 받아, 병합 파라미터 공간과 최적화할 평가 지표를 정의합니다. 기본 구조는 다음과 같습니다:

```yml
genome:
    models:
       - model_1
       - model_2
       ...
       - model_n
    merge_method: dare_ties
    base_model: base_model_if_needed
    tokenizer_source: null # optional
    layer_granularity: 8

    # optional:
    normalize: false
    allow_negative_weights: false
    smooth: false
    filters: ...
tasks:
  - name: lm_eval_task_name
    weight: 1.0 # optional
    metric: "acc,none" # defaults to acc,none
  - name: ... # as many as you want
```

---

## Genome Definition

`genome` 섹션은 `mergekit-evolve`가 탐색할 **파라미터 공간(parameter space)**을 정의합니다.

### `models`

병합에 사용할 수 있는 모든 모델의 리스트입니다. 병합 방법에 따라 최종 결과에 모든 모델이 반드시 포함되지는 않을 수 있습니다.

### `merge_method`

사용할 병합 방법입니다. 현재 지원되는 값은 다음과 같습니다:

* `linear`
* `dare_ties`
* `task_arithmetic`
* `ties`
* `slerp`

### `base_model`

필요한 경우 병합의 기준이 되는 base model을 지정합니다.

### `layer_granularity`

모델 레이어를 일정 크기의 블록으로 나누어, **블록 단위로 서로 다른 병합 파라미터**를 학습하도록 합니다.

예시:

* 32-layer 모델
* `layer_granularity: 8`
  → 8개 레이어씩 4개 그룹

이 값은 **모델 레이어 수의 약수(divisor)**여야 합니다.

* 값이 클수록 탐색 공간이 작아져 **수렴 속도는 빨라지지만**, 전역 최적해를 놓칠 가능성이 있습니다.
* 설정하지 않으면 모든 레이어에 동일한 파라미터를 사용합니다.

### `normalize`

병합 시 `normalize` 플래그를 설정합니다.

* `linear`, `ties`, `dare_ties`와 같은 방법에서
* 항상 **유효한 모델 공간**만 탐색하도록 제약을 걸어줍니다.

이는 `layer_granularity`와 마찬가지로 수렴 속도를 크게 향상시킬 수 있지만, 비정형이지만 성능이 좋은 해를 배제할 수 있습니다.

### `allow_negative_weights`

말 그대로 **음수 weight 허용 여부**입니다.

* 설정하지 않으면 weight의 절댓값이 사용됩니다.
* `linear`, `slerp`에서는 탐색 공간 축소에 유리합니다.
* **task arithmetic 계열**에서는 보통 `true`로 설정하는 것이 좋습니다.

### `smooth`

`true`로 설정하면 레이어 블록 간 파라미터를 **보간(interpolation)**합니다.

* `false`: 각 블록이 고정된 값 사용
* `true`: 레이어에 따라 부드럽게 변화

### `filters`

`mergekit-yaml`과 동일한 **filter 메커니즘**을 사용하여 파라미터를 분리할 수 있습니다.

예시 (LLaMA 계열 모델):

```yaml
filters:
  - self_attn
  - mlp
```

이 경우 파라미터 공간이 다음과 같이 나뉩니다:

1. Self-attention 파라미터
2. MLP 파라미터
3. 그 외 나머지

프롬프트 포맷이 다른 모델을 병합할 때 매우 유용하지만, **파라미터 차원이 크게 증가**합니다.

---

## Task Definition

병합 결과를 평가하기 위해 EleutherAI **LM Evaluation Harness**에서 지원하는 task 목록을 정의해야 합니다.

* [Built-in tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)
* 사용자 정의 task (권장)

  * [New Task Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)

기본 metric은 `acc`입니다. 다른 metric을 사용하는 경우 반드시 명시해야 합니다.

각 task는 선택적으로 **weight**를 가질 수 있습니다.

⚠️ `mergekit-evolve`는 **점수를 최대화(maximize)**합니다.

* Perplexity처럼 **낮을수록 좋은 지표**는 반드시 **음수 weight**를 사용하세요.

---

## Running `mergekit-evolve`

```sh
mergekit-evolve [OPTIONS] --storage-path PATH GENOME_CONFIG_PATH
```

`--storage-path`에는 다음이 저장됩니다:

* 입력 모델
* 평가 중인 병합 결과
* 현재 최고 성능 병합의 설정 파일

⚠️ 디스크 기반 병합 시 **GPU당 fp16 모델 1개 이상 용량**이 필요할 수 있습니다.

---

## Scheduling Strategy (`--strategy`)

### `pool` (권장 기본값)

* GPU 하나당 actor 하나 할당
* 병합과 평가를 동일 노드에서 수행
* 단일 / 분산 환경 모두 안정적

### `buffered`

* 항상 GPU마다 평가 대기 모델을 유지
* 병합과 평가를 **동시에 수행 가능**
* 단일 노드 또는 빠른 공유 파일시스템에서만 사용 권장

### `serial`

* Ray placement group 사용
* 나머지는 Ray에 전적으로 위임
* 다른 전략이 잘 안 될 때만 시도

---

## Evaluation LLM Backend

* 기본값: HuggingFace (`hf`)
* vLLM 사용 시 `--vllm` 플래그 추가

---

## On-Disk vs. In-Memory

기본 동작:

1. 병합 수행
2. 디스크에 저장
3. lm-eval 실행

→ 안정적이지만 **느리고 디스크 사용량 큼**

`pool` 전략에서는 **in-memory 병합**이 가능합니다:

* 디스크 저장 없이
* vLLM 내부 파라미터를 직접 갱신
* 매우 빠르고 디스크 사용 없음
* ⚠️ 내부 구현에 의존 → 언제든 깨질 수 있음

활성화:

```sh
--in-memory
```

---

## Task Search Path

커스텀 task를 사용하는 경우 검색 경로를 추가할 수 있습니다:

```sh
--task-search-path /path/to/tasks
```

여러 번 지정 가능

---

## Batch Size

평가 시 batch size를 오버라이드합니다.

* vLLM 사용 시 `auto` 권장 (기본값)

---

## CMA-ES Options

### `--max-fevals`

* 평가할 병합 수의 최대값
* 기본값: 100
* CMA-ES 특성상 **최대 50% 초과**될 수 있음

### `--sigma0`

* CMA-ES 초기 sigma 값
* 특별한 경우가 아니면 조정할 필요 없음

---

## WandB Logging

Weights & Biases 로깅 지원:

```sh
--wandb
--wandb-project <project>
--wandb-entity <entity>
```

---

## Example

```sh
mergekit-evolve \
  --strategy pool \
  --wandb \
  --wandb-project mergekit-evolve \
  --wandb-entity arcee-ai \
  --storage-path /path/to/mergekit-evolve/ \
  ./config.yml
```

---

## Output

* 현재까지 최고 성능 병합 설정이 다음 파일로 저장됩니다:

```text
best_config.yaml
```

* WandB 사용 시 config가 artifact로도 저장됩니다.
* `Ctrl+C` 또는 `--max-fevals` 초과 시 종료됩니다.

---

## Caveats

`mergekit-evolve`는 아직 **활발히 개발 중**이며, 모든 환경에서 충분히 테스트되지 않았을 수 있습니다.

* 실행 초기에 로그를 꼭 확인하세요
* 문제가 있으면 GitHub Issue 제출을 권장합니다

---

## Acknowledgements

* SakanaAI: 아이디어 제공
* EleutherAI: LM Evaluation Harness


# mergekit-evolve

`mergekit-evolve` is a script that uses an evolutionary algorithm (CMA-ES) to optimize the parameters of a merge against model metrics. This is inspired by SakanaAI's [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187), in particular their parameter-space approach. `mergekit-evolve` uses EleutherAI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to define and evaluate the scoring function. The script is set up to be run either single-node or on a Ray cluster and has a few different strategies for scheduling operations depending on your particular configuration of compute.

## Installation

Install `mergekit` with the `evolve` (and optionally `vllm`) features:

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .[evolve,vllm]
```

If you had a perfectly good pytorch environment going and installing an older version of vLLM downgraded it and broke flash attention, run the following commands to fix it:

```sh
pip uninstall flash-attn
pip cache purge
pip install flash-attn
```

## Configuration

`mergekit-evolve` takes in a YAML configuration file that defines how the merge is parameterized and what metrics to optimize. The general syntax is as follows:

```yml
genome:
    models:
       - model_1
       - model_2
       ...
       - model_n
    merge_method: dare_ties
    base_model: base_model_if_needed
    tokenizer_source: null # optional
    layer_granularity: 8

    # optional:
    normalize: false
    allow_negative_weights: false
    smooth: false
    filters: ...
tasks:
  - name: lm_eval_task_name
    weight: 1.0 # optional
    metric: "acc,none" # defaults to acc,none
  - name: ... # as many as you want
```

### Genome Definition

The `genome` section of the configuration file defines the parameter space that `mergekit-evolve` will be optimizing in.

#### `models`

This should be a list of all of the models you want available to be merged. Depending on the merge method not all are guaranteed to be used in the final merge.

#### `merge_method`

Merge method to be used. Currently supported values are `linear`, `dare_ties`, `task_arithmetic`, `ties`, and `slerp`.

#### `base_model`

The base model for the merge, if applicable.

#### `layer_granularity`

A set of parameters will be introduced for each consecutive slice of `layer_granularity` layers. So for example, a 32-layer model like `mistralai/Mistral-7B-v0.1` with `layer_granularity: 8` will be divided into 4 groups of 8 layers with different merge parameters for each. The value specified here must be a divisor of the number of layers in your input models. Large values of `layer_granularity` will reduce the search space greatly, meaning you will get faster convergence at the cost of a potentially less good global solution.

When not set, one set of parameters will be used for all layers.

#### `normalize`

Sets the `normalize` flag when merging. For methods like `linear`, `ties`, and `dare_ties` this constrains the search space to a set of definitely valid models. Similarly to `layer_granularity`, this can greatly speed up convergence at the cost of ruling out oddball solutions that might score better than more standard merges.

#### `allow_negative_weights`

Pretty self explanatory. When this flag is not set, the absolute value of weight parameters is used. Sensible search space reduction for `linear` and `slerp`. For task arithmetic based methods you probably want `allow_negative_weights: true`.

#### `smooth`

If set to `true`, then parameter values will be interpolated across layers instead of assigning a single, fixed value to each block.

#### `filters`

Accepts a list of filters, as in `mergekit-yaml`, by which to separate the parameters. So, for example, setting filters as below for a Llama-based merge:

```yaml
filters:
  - self_attn
  - mlp
```

Will divide up the merge parameters into three groups - self attention parameters, MLP parameters, and a third for everything else. Separating the parameters out like this can be very beneficial when merging models trained on different prompt formats. It also makes your parameter space three times as big though!

### Task Definition

To evaluate the produced merges you need to specify a list of tasks supported by the EleutherAI LM evaluation harness. This can be either [built in tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks) (don't be naughty) or tasks you define yourself (see the [New Task Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) for how). If your task does not use `acc` as the metric then you must specify the correct metric name. Each task can also optionally have a weight associated.

`mergekit-evolve` aims to maximize the score of the merge, so if you are using any tasks or metrics where a lower score is better (like perplexity) be sure to assign a negative weight to that task.

## Running `mergekit-evolve`

```sh
mergekit-evolve [OPTIONS] --storage-path PATH GENOME_CONFIG_PATH
```

`mergekit-evolve` needs a storage path specified, where it will save the input models, merges to evaluate, and the config for the current best merge evaluated. If you are not using in-memory merging this can require a _lot_ of space - expect at least one fp16 model per GPU.

Some important options:

### Scheduling Strategy (`--strategy`)

There are three different strategies implemented for scheduling merging and evaluation jobs.

#### `pool`

Assigns an actor to each GPU in your cluster and guarantees merges and evaluations are performed on the same node. This is a safe default suitable for any configuration, local or distributed.

#### `buffered`

Maintains a buffer of tasks scheduled to ensure that there is always a model merging or ready to evaluate for each GPU. Allows for concurrent merging and evaluation of models on the same GPU if enough VRAM is available. Only suitable for a single-node setup or when `--storage-path` points to a fast shared filesystem.

#### `serial`

Uses Ray placement groups to ensure merges and their evaluations happen on the same node, but otherwise just lets Ray take the wheel. Maybe give a try if you're having trouble with the other two, otherwise probably don't use it.

### Evaluation LLM Backend

By default `mergekit-evolve` will use the `hf` backend for `lm-eval`. To use vLLM instead, pass the `--vllm` flag.

### On-Disk vs. In-Memory

By default `mergekit-evolve` will perform merges, write the result to disk, then start up an instance of lm-eval pointing at that path. This is a safe default and will generally always work but also causes a lot of GPU downtime and eats disk space. When using the `pool` scheduling strategy, you have the option to instead keep a model resident in memory and directly update its parameters instead of merging to disk. This is much faster and uses no additional disk space. However, it does involve mucking around in the internals of vLLM and the LM evaluation harness. So it might break at any moment! Choose wisely. Use `--in-memory` to enable this mode.

### Task search path

If you're using custom task definitions (and you should be) then you can append to the search path using the `--task-search-path` option. This should point to the directory your custom task YAML is in (or a parent of that directory). Multiple paths can be included by repeating the option.

### Batch size

Override the batch size used during merge evaluation. If using vLLM `auto` is recommended (default).

### CMA-ES options

#### `--max-fevals`

Maximum number of merges to evaluate. Note that the `cma` package is very loosey-goosey with this number and will happily go over by 50% depending on the size of each generation. Set to 100 by default.

#### `--sigma0`

Initial value of sigma for CMA-ES. No need to play with this unless you really know what you're doing.

### WandB logging

`mergekit-evolve` supports logging metrics to Weights & Biases. Enable this functionality with the `--wandb` flag. Project and entity names can be overridden with the `--wandb-project` and `--wandb-entity` options.

### Example

```sh
mergekit-evolve --strategy pool --wandb --wandb-project mergekit-evolve --wandb-entity arcee-ai --storage-path /path/to/mergekit-evolve/ ./config.yml
```

## Output

`mergekit-evolve` will write the merge configuration for the best merge found so far to the storage path with the filename `best_config.yaml`. If you're using WandB it will also log the config as an artifact. The script will keep running until a KeyboardInterrupt is received or `--max-fevals` is generously exceeded.

## Caveats

`mergekit-evolve` is a work in progress and has probably not been tested on your specific configuration. Keep an eye on the output before leaving it running, and if you run in to any issues don't hesitate to file an issue!

## Acknowledgements

Thanks to SakanaAI for the inspiration and the EleutherAI team for the LM evaluation harness.
