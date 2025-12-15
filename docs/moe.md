# mergekit-moe

`mergekit-moe`는 **동일한 크기의 Mistral 또는 Llama 계열 모델들**을 결합하여 **Mixtral-style Mixture of Experts (MoE)** 모델을 생성하는 스크립트입니다. 이 스크립트는 다음과 같은 방식으로 동작합니다:

* **Self-Attention** 및 **LayerNorm** 파라미터는 하나의 **base model**에서 가져오고
* **MLP 파라미터**는 여러 개의 **expert models**에서 가져옵니다.

`hidden` 또는 `cheap_embed` **gate mode**를 사용할 경우, **추가 학습 없이 바로 사용 가능한 모델**이 생성됩니다. 반대로 **sparse upcycling** 등 **추가 학습을 전제로 한 초기화**가 목적이라면 `random` gate mode를 사용하세요.

---

## Configuration

`mergekit-moe`는 자체적인 YML 설정 문법을 사용합니다. 기본 형태는 다음과 같습니다:

```yml
base_model: path/to/self_attn_donor
gate_mode: hidden # one of "hidden", "cheap_embed", or "random"
dtype: bfloat16 # output dtype (float32, float16, or bfloat16)
## (optional)
# experts_per_token: 2
experts:
  - source_model: expert_model_1
    positive_prompts:
      - "This is a prompt that is demonstrative of what expert_model_1 excels at"
    ## (optional)
    # negative_prompts:
    #   - "This is a prompt expert_model_1 should not be used for"
  - source_model: expert_model_2
  # ... and so on
```

스크립트 실행 방식은 다음과 같습니다:

```bash
mergekit-moe ./config.yml ./my-clowncar-moe-12x180B
```

---

## Supported Architectures

현재 `mergekit-moe`는 다음 MoE 아키텍처를 출력할 수 있습니다:

* **Mixtral**
* **DeepSeek MoE**
* **Qwen MoE**

일부 아키텍처는 **shared expert**를 지원하며, 이는 모든 토큰에서 항상 활성화되는 expert입니다. 설정 예시는 다음과 같습니다:

```yml
base_model: path/to/self_attn_donor
gate_mode: hidden
dtype: bfloat16
experts:
  ...
shared_experts:
  - source_model: model_name
    positive_prompts: # Qwen MoE + hidden gate mode에서 필수
      - "blah blah"
    # (optional, but recommended)
    residual_scale: 0.1 # shared expert 출력의 과도한 영향 방지
```

> ⚠️ 현재 **shared expert는 최대 1개만 지원**됩니다.

---

## Architecture Inference

입력 모델과 `shared_experts` 설정 여부에 따라 **적절한 MoE 아키텍처가 자동으로 추론**됩니다.

원할 경우, 아래와 같이 `architecture:` 필드를 통해 **명시적으로 지정**할 수도 있습니다:

```yml
base_model: path/to/self_attn_donor
architecture: qwen
# ...
```

---

## Gate Modes

MoE gate 파라미터를 초기화하는 방식으로, 현재 3가지 모드가 구현되어 있습니다.

### `hidden` (default, 권장)

* **Positive / Negative prompt**를 base model에 실제로 forward하여
* **hidden state representation**을 사용해 gate 파라미터를 설정
* **가장 품질이 좋고 효과적**
* 단점: **GPU 메모리 사용량 큼**

> 메모리 제약이 있다면 `--load-in-8bit` 또는 `--load-in-4bit` 옵션 사용 가능

---

### `cheap_embed`

* prompt의 **token embedding만 사용**
* 모든 layer에서 **동일한 gate 파라미터** 사용
* `hidden` 대비 **품질은 낮지만**, **아주 저사양 환경에서도 실행 가능**

---

### `random`

* gate 파라미터를 **무작위 초기화**
* **추가 파인튜닝 전제** (sparse upcycling 등)
* 또는 "약간 미친 모델"을 원할 때(?)도 가능

---

## Example Configurations

### Example 1: smol_llama Sparse Upcycling (8×220M MoE)

```yml
base_model: BEE-spoke-data/smol_llama-220M-GQA
gate_mode: random
dtype: bfloat16
experts:
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
# 이후 학습 진행
```

---

### Example 2: Mistral Models "Clown Car" MoE

```yml
base_model: NousResearch/Hermes-2-Pro-Mistral-7B
gate_mode: hidden
dtype: bfloat16
experts:
  - source_model: NousResearch/Hermes-2-Pro-Mistral-7B
    positive_prompts:
      - "<|im_start|>user\nHello, who are you?<|im_end|>"
      - "<|im_start|>user\nI need help with"
  - source_model: BioMistral/BioMistral-7B-DARE
    positive_prompts:
      - "As a doctor of medicine,"
  - source_model: PocketDoc/Dans-AdventurousWinds-7b
    positive_prompts:
      - "[Genres: Science Fiction]\n[Tags: humor, old school, sci fi]"
      - "> get ye flask"
      - "[Mode: Interactive Storyteller]"
  - source_model: VAGOsolutions/SauerkrautLM-7b-HerO
    positive_prompts:
      - "<|im_start|>user\nWie geht es dir?<|im_end|>"
      - "Das ist ein Satz auf Deutsch."
```

---

## FAQ

### Q. "Your model has duplicated tensors but the --clone-tensors flag is not set" 경고의 의미는?

**Charles O. Goddard (cg123)**의 설명:

> 이는 **완전히 무해한 경고**입니다. 하나의 tensor가 여러 위치에서 재사용될 때 발생합니다.
> 예를 들어:
>
> * MoE sparse upcycling
> * passthrough merge에서 layer 반복 사용
>
> `--clone-tensors`를 켜면 메모리는 조금 더 사용하지만,
> 저장 속도 저하 및 일시적 메모리 스파이크를 줄일 수 있습니다.
>
> 차이가 매우 작기 때문에, 이 경고 자체를 제거해도 될 정도입니다.

(관련 GitHub 이슈 참고: #279)


# mergekit-moe

`mergekit-moe` is a script for combining Mistral or Llama models of the same size into Mixtral Mixture of Experts models. The script will combine the self-attention and layer normalization parameters from a "base" model with the MLP parameters from a set of "expert" models.

If using the `hidden` or `cheap_embed` gate mode, the output model will be usable without any further training. If you are initializing a model to do further training on, such as for sparse upcycling, then use the `random` gate mode to get a model ready for training.

## Configuration

`mergekit-moe` uses its own YML configuration syntax, which looks like so:

```yml
base_model: path/to/self_attn_donor
gate_mode: hidden # one of "hidden", "cheap_embed", or "random"
dtype: bfloat16 # output dtype (float32, float16, or bfloat16)
## (optional)
# experts_per_token: 2
experts:
  - source_model: expert_model_1
    positive_prompts:
      - "This is a prompt that is demonstrative of what expert_model_1 excels at"
    ## (optional)
    # negative_prompts:
    #   - "This is a prompt expert_model_1 should not be used for"
  - source_model: expert_model_2
  # ... and so on
```

The script takes two arguments, an input config and an output path: `mergekit-moe ./config.yml ./my-clowncar-moe-12x180B`

Currently the script can output models that use the Mixtral, Deepseek MoE, or Qwen MoE architectures. Some output architectures support a shared expert which will be activated for all tokens, which can be configured like this:

```yml
base_model: path/to/self_attn_donor
gate_mode: hidden # one of "hidden", "cheap_embed", or "random"
dtype: bfloat16 # output dtype (float32, float16, or bfloat16)
experts:
  ...
shared_experts:
  - source_model: model_name
    positive_prompts: # required by Qwen MoE for "hidden" gate mode, otherwise not allowed
      - "blah blah"
    # (optional, but recommended:)
    residual_scale: 0.1 # downweight output from shared expert to prevent overcooking the model
```

Currently only up to one shared expert is supported.

An appropriate architecture will be inferred based on the input models and presence or absence of shared experts in your configuration. Alternatively, you can explicitly specify an output architecture by setting the `architecture:` field in your config. For example:

```yml
base_model: path/to/self_attn_donor
architecture: qwen
# ... and so on
```

### Gate Modes

There are three methods for populating the MoE gates implemented.

#### "hidden"

Uses the hidden state representations of the positive/negative prompts for MoE gate parameters. Best quality and most effective option; the default. Requires evaluating each prompt using the base model so you might not be able to use this on constrained hardware (depending on the model). You can use `--load-in-8bit` or `--load-in-4bit` to reduce VRAM usage.

#### "cheap_embed"

Uses only the raw token embedding of the prompts, using the same gate parameters for every layer. Distinctly less effective than "hidden". Can be run on much, much lower end hardware.

#### "random"

Randomly initializes the MoE gates. Good for if you are going to fine tune the model afterwards, or maybe if you want something a little unhinged? I won't judge.

## Example Configurations

Sparse upcycling of smol_llama into a 8x220M MoE:

```yml
base_model: BEE-spoke-data/smol_llama-220M-GQA
gate_mode: random
dtype: bfloat16
experts:
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
# and then train the sucker!
```

Shove some Mistral models in a clown car:

```yml
base_model: NousResearch/Hermes-2-Pro-Mistral-7B
gate_mode: hidden
dtype: bfloat16
experts:
  - source_model: NousResearch/Hermes-2-Pro-Mistral-7B
    positive_prompts:
      - "<|im_start|>user\nHello, who are you?<|im_end|>"
      - "<|im_start|>user\nI need help with"
  - source_model: BioMistral/BioMistral-7B-DARE
    positive_prompts:
      - "As a doctor of medicine,"
  - source_model: PocketDoc/Dans-AdventurousWinds-7b
    positive_prompts:
      - "[Genres: Science Fiction]\n[Tags: humor, old school, sci fi]"
      - "> get ye flask"
      - "[Mode: Interactive Storyteller]"
  - source_model: VAGOsolutions/SauerkrautLM-7b-HerO
    positive_prompts:
      - "<|im_start|>user\nWie geht es dir?<|im_end|>"
      - "Das ist ein Satz auf Deutsch."
```

## FAQ

### What does the "Your model has duplicated tensors but the --clone-tensors flag is not set" warning mean?

Answer from [Charles O. Goddard (cg123)](https://github.com/cg123)
(also see [this GitHub issue](https://github.com/arcee-ai/mergekit/issues/279#issuecomment-2081818104)):

> This is completely benign. This happens when a single tensor from a model is used in multiple places, like when doing sparse upcycling with the moe script or doing passthrough merges that repeat layers. Having `--clone-tensors` set can use slightly more memory, but having it unset will slow down saving and introduce small memory usage spikes in cases where this warning occurs. It's honestly a small enough difference that the warning could be removed entirely.
