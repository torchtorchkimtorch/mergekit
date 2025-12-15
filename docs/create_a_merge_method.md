# Custom Merge Method로 MergeKit 확장하기

## 개요

MergeKit은 **커스텀 merge method**를 구현하기 위한 두 가지 경로를 제공합니다:

|             | 데코레이터 API     | 클래스 기반 API        |
| ----------- | ------------- | ----------------- |
| **복잡도**     | 간단한 함수 기반     | 전체 클래스 구현         |
| **추상화 수준**  | 높음            | 낮음                |
| **파라미터 처리** | 자동 검증         | 수동 설정             |
| **실행 흐름**   | 단일 함수         | 임의의 계산 그래프        |
| **적합한 경우**  | 대부분의 merge 방법 | 다단계·다중 입력의 복잡한 전략 |

두 접근 방식 모두 MergeKit의 **태스크 시스템**을 활용하여 리소스 관리와 실행 제어의 이점을 공유합니다. 어떤 방식을 선택할지는 merge 연산의 복잡도와 필요한 제어 수준에 따라 달라집니다.

**파라미터 설정에 대한 참고:** MergeKit은 **계층적 YAML 기반 설정 시스템**을 사용합니다. 커스텀 merge method의 파라미터(스칼라/모델별)는 전역, 모델별, 슬라이스별 등 여러 수준에서 정의할 수 있습니다. 실제로 함수나 태스크에 전달되는 값은 이 계층과 컨텍스트를 기반으로 MergeKit이 해석·해결합니다. 설정 구조와 우선순위에 대한 자세한 내용은 README의 *Merge Configuration* 섹션을 참고하세요.

### 핵심 태스크 시스템 기능

MergeKit의 계산 그래프 인프라는 모든 merge method가 상속하는 고급 리소스 관리 기능을 제공합니다:

* **스마트 메모리 관리**

  * 반환값 생명주기 자동 추적
  * 더 이상 필요 없는 값의 조기 제거
  * 태스크 그룹에 따른 shard 로딩 최적화

* **디바이스 관리**

  * 연산/저장 디바이스 간 텐서 자동 이동
  * CPU 및 GPU 실행 모두 지원

* **태스크 스케줄링**

  * 메모리 사용 최소화를 위한 shard 기준 태스크 그룹화
  * 우선순위 시스템을 통한 로딩 지연
  * shard 상주(residency)를 최적화하는 실행 순서

### 데코레이터 API

단일 텐서 변환으로 표현 가능한 **단순한 merge 연산**에 적합합니다. 주요 특징:

* 파라미터 검증, 타입 체크, 값 해석
* 설정 스키마 자동 생성
* 베이스 모델 처리 단순화
* 기본 GPU 가속 옵트인

### 클래스 기반 API

다음이 필요한 경우 선택하세요:

* 다단계 merge 연산
* 커스텀 계산 그래프
* 가중치 메타데이터에 대한 직접 접근
* 복잡한 파라미터 타입
* 실행에 대한 세밀한 제어

## 데코레이터 API 구현

### 기본 워크플로우

1. merge 로직을 담은 **타입 주석이 있는 Python 함수** 정의
2. 설정과 함께 `@merge_method` 데코레이터 추가
3. 해당 함수를 포함한 모듈을 **import**

   * MergeKit이 데코레이터 기반 merge method를 발견하려면, 그 함수가 정의된 Python 모듈이 초기화 시 import되어야 합니다. `mergekit/merge_methods/__init__.py`에 해당 모듈을 import하세요. 모듈이 import되면 `@merge_method` 데코레이터가 자동으로 메서드를 등록합니다.

### 예시: 가중 평균 (Weighted Average)

```python
from mergekit.merge_methods.easy_define import merge_method
from typing import List
import torch

@merge_method(
    name="weighted_average",
    pretty_name="Weighted Average",            # 선택: 사람이 읽기 쉬운 이름
    reference_url="https://example.com/docs",  # 선택: 문서 또는 논문 링크
)
def average_merge(
    tensors: List[torch.Tensor],  # 필수: 입력 텐서
    weight: List[float],          # 벡터 파라미터 (모델당 하나)
    normalize: bool = True,       # 기본값이 있는 스칼라 파라미터
) -> torch.Tensor:
    if normalize:
        total = sum(weight)
        weight = [w / total for w in weight]

    return sum(t * w for t, w in zip(tensors, weight))
```

다음과 같은 설정이 가능합니다:

```yaml
merge_method: weighted_average
models:
  - model: model1
    parameters:
      weight: 0.3
  - model: model2
    parameters:
      weight: 0.7
parameters: # 전역 파라미터
  normalize: true
```

### 파라미터 타입과 처리

데코레이터는 세 가지 파라미터 범주를 지원합니다:

1. **스칼라 파라미터**

   * 타입: `bool`, `float`, `int`
   * 모든 모델에 공통으로 적용
   * 기본값이 없으면 필수 파라미터
   * 예: `normalize: bool = True`

2. **벡터 파라미터**

   * 타입: `List[float]` 또는 `List[int]`
   * 모델별로 설정
   * 기본값은 리스트가 아닌 **단일 숫자**여야 하며, 모델 수에 맞게 브로드캐스트됨
   * 예: `weights: List[float]`

3. **베이스 모델 통합**
   함수 시그니처의 `tensors: List[torch.Tensor]`와 선택적 `base_tensor`는 YAML의 `base_model` 설정과 다음과 같이 상호작용합니다:

   * **함수에 `base_tensor` 파라미터가 있는 경우** (예: `base_tensor: torch.Tensor` 또는 `Optional[torch.Tensor]`):

     * `base_model`이 지정되면 해당 텐서가 `base_tensor`로 전달됩니다. `Optional`로 주석되어 있고 `base_model`이 없으면 `None`입니다.
     * `tensors`에는 `models:`에 나열된 모델의 텐서만 순서대로 포함되며, 베이스 모델 텐서는 포함되지 않습니다.
   * **함수에 `base_tensor`가 없는 경우**:

     * `base_model`이 지정되면 그 텐서는 `tensors[0]`에 포함됩니다.
     * 이후 `tensors[1:]`은 `models:`에 나열된 모델 순서와 일치합니다.
     * `base_model`이 없으면 `tensors`는 `models:`와 1:1 대응합니다.

4. **자동 주입 특수 파라미터**
   특정 이름의 파라미터는 YAML로 설정하지 않으며, MergeKit이 자동으로 주입합니다:

   * `output_weight: WeightInfo` — 계산 중인 가중치 텐서의 메타데이터
   * `base_model: ModelReference` (또는 `Optional[ModelReference]`) — 설정에 베이스 모델이 있을 경우 해당 참조

## 클래스 기반 API 구현

세밀한 제어가 필요한 복잡한 merge의 경우 `MergeMethod`와 `Task` 클래스를 구현합니다.

### 구현 예시

```python
from mergekit.merge_methods.base import MergeMethod, ConfigParameterDef
from mergekit.common import ImmutableMap, ModelReference, WeightInfo
from mergekit.graph import Task
from typing import Any, Dict, List
import torch


class CustomDependencyTask(Task[float]):
    totally_real_parameter: str

    def execute(self) -> float:
        return 42.0

class CustomMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    parameters: ImmutableMap[str, Any]
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {
            "tensors": self.gather_tensors,
            "dependency": CustomDependencyTask(totally_real_parameter="example"),
        }

    def priority(self) -> int:
        return 1

    def group_label(self) -> str:
        return self.weight_info.name

    def uses_accelerator(self) -> bool:
        return True

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], dependency: float
    ) -> torch.Tensor:
        result = ...
        return result


class CustomMerge(MergeMethod):
    def name(self) -> str:
        return "custom_merge"

    def pretty_name(self) -> str:
        return "Custom Merge"

    def reference_url(self) -> str:
        return "https://example.com/custom"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef("threshold", float, required=False, default_value=0.5)
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef("weight", float, required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task:
        return CustomMergeTask(
            gather_tensors=tensors,
            parameters=parameters,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
        )
```

### 태스크 스케줄링 시스템

클래스 기반 API는 실행 제어에 대한 세밀한 옵션을 제공합니다:

* **우선순위 제어**: `priority()`로 실행 순서 영향
* **태스크 그룹화**: `group_label()`로 유사 연산 배치
* **리소스 관리**:

  * 텐서 생명주기 자동 추적
  * 조기 제거를 통한 메모리 최적화
  * 연산/저장 디바이스 자동 배치
* **계산 그래프**: 여러 태스크를 연결해 복잡한 흐름 구성

### 구현 요구사항

1. **Task 클래스**

   * 타입 주석이 있는 `execute()` 구현
   * 의존성 선언을 위한 `arguments()` 구현
   * 선택적으로 `priority()`, `group_label()`, `uses_accelerator()` 오버라이드

2. **Method 클래스**

   * 필수 메서드: `name()`, `make_task()`
   * 선택 메서드: `pretty_name()`, `reference_url()`
   * `parameters()` 및 `tensor_parameters()`로 파라미터 정의

### 등록 방법

클래스 기반 메서드는 `mergekit/merge_methods/registry.py`의 `STATIC_MERGE_METHODS`에 추가합니다:

```python
from mergekit.merge_methods.my_module import CustomMerge

STATIC_MERGE_METHODS: List[MergeMethod] = [
    CustomMerge(),
]
```

## 참고 구현

1. **Linear Merge** (`mergekit.merge_methods.linear`)

   * 기본 가중 평균
   * 클래스 기반 구현의 좋은 예시

2. **Multi-SLERP** (`mergekit.merge_methods.multislerp`)

   * 초구면 보간
   * 데코레이터 API의 복잡한 사용 예시

3. **Task Arithmetic** (`mergekit.merge_methods.task_arithmetic`)

   * 고급 그래프 기반 구현
   * TIES/크기 기반 프루닝 예시


# Extending MergeKit with Custom Merge Methods

## Overview

MergeKit offers two different paths for implementing custom merge methods:

|                        | Decorator API         | Class-based API                                |
| ---------------------- | --------------------- | ---------------------------------------------- |
| **Complexity**         | Simple function-based | Full class implementation                      |
| **Abstraction Level**  | Higher-level          | Lower-level                                    |
| **Parameter Handling** | Automatic validation  | Manual configuration                           |
| **Execution Flow**     | Single function       | Arbitrary computation graph                    |
| **Best For**           | Most merge methods    | Complex multi-stage, multi-input strategies    |

Either approach benefits from MergeKit's underlying task system for resource management and execution control. The question of which to use largely depends on the complexity of the merge operation and the level of control needed.

**Note on Parameter Configuration:** MergeKit uses a hierarchical YAML-based configuration system. Parameters for your custom merge methods (both scalar and per-model) can be defined at various levels (e.g., globally, per-model, per-slice). The values your merge function or task receives are resolved by MergeKit based on this hierarchy and context. For full details on configuration structure and parameter precedence, please refer to the [Merge Configuration](../README.md#merge-configuration) section of the README.

### Core Task System Features

MergeKit's computational graph infrastructure provides sophisticated resource management that all merge methods inherit:

- **Smart Memory Management**
  - Automatic return value lifecycle tracking
  - Early value eviction when no longer needed
  - Optimized shard loading based on task groups

- **Device Management**
  - Automatic tensor movement between compute and storage devices
  - Support for both CPU and GPU execution

- **Task Scheduling**
  - Tasks grouped by tensor shard to minimize memory usage
  - Loads deferred until last possible moment (via priority system)
  - Execution ordered to optimize shard residency

### Decorator API

Best for straightforward merge operations that can be expressed as a single tensor transformation. Features:

- Parameter validation, type checking, and value resolution
- Configuration schema generation
- Simplified base model handling
- Default GPU acceleration opt-in

### Class-based API

Choose when you need:

- Multi-stage merge operations
- Custom computation graphs
- Direct access to weight metadata
- Complex parameter types
- Fine-grained control over execution

## Decorator API Implementation

### Basic Workflow

1. Define a type-annotated Python function with your merge logic
2. Add the `@merge_method` decorator with configuration
3. Ensure the module containing your function is imported
   - For MergeKit to discover your decorated merge method, the Python module containing it must be imported during MergeKit's initialization. Add an import statement for your module in `mergekit/merge_methods/__init__.py`. Once the module is imported, the `@merge_method` decorator handles the registration of the method with MergeKit.

### Example: Weighted Average

```python
from mergekit.merge_methods.easy_define import merge_method
from typing import List
import torch

@merge_method(
    name="weighted_average",
    pretty_name="Weighted Average",            # Optional: human-readable name
    reference_url="https://example.com/docs",  # Optional: documentation or paper link
)
def average_merge(
    tensors: List[torch.Tensor],  # Required: input tensors
    weight: List[float],          # Vector parameter (one float per model)
    normalize: bool = True,       # Scalar parameter with default
) -> torch.Tensor:
    if normalize:
        total = sum(weight)
        weight = [w / total for w in weight]

    return sum(t * w for t, w in zip(tensors, weight))
```

This enables configurations like:

```yaml
merge_method: weighted_average
models:
  - model: model1
    parameters:
      weight: 0.3
  - model: model2
    parameters:
      weight: 0.7
parameters: # Global parameters
  normalize: true
```

### Parameter Types and Handling

The decorator supports three parameter categories:

1. **Scalar Parameters**
   - Types: `bool`, `float`, or `int`
   - Single value for all models
   - Without defaults they become required parameters
   - Example: `normalize: bool = True`

2. **Vector Parameters**
   - Types: `List[float]` or `List[int]` only
   - Configured per-model
   - Default values must be single numbers, not lists, as they are broadcasted
   - Example: `weights: List[float]`

3. **Base Model Integration**
    The `tensors: List[torch.Tensor]` argument and an optional `base_tensor` argument in your function signature interact with the `base_model` specified in the YAML configuration as follows:

    - **If your function includes a `base_tensor` parameter (e.g., `base_tensor: torch.Tensor` or `base_tensor: Optional[torch.Tensor]`):**
        - The `base_tensor` argument will receive the tensor from the `base_model` specified in the YAML. If annotated as `Optional` and no `base_model` is configured, it will be `None`.
        - The `tensors: List[torch.Tensor]` argument will *only* contain tensors from the models specified under the `models:` key in the YAML, in order. It will *not* include the base model's tensor.
    - **If your function does *not* include a `base_tensor` parameter:**
        - If a `base_model` is specified in the YAML, its tensor will be the *first element* in the `tensors: List[torch.Tensor]` list (i.e., `tensors[0]`).
        - Subsequent elements (`tensors[1:]`) will correspond to the models listed under the `models:` key in the YAML, in order.
        - If no `base_model` is specified, the `tensors` list will directly correspond to the models listed under the `models:` key.

4. **Special Auto-Populated Parameters**
    Certain parameter names in your function signature have special meaning and are auto-populated by MergeKit if present. You do not configure these directly in the YAML `parameters` sections for your method; MergeKit provides them.
    - `output_weight: WeightInfo`: If your function accepts an argument named `output_weight` annotated with `WeightInfo`, MergeKit will pass metadata about the specific weight tensor being computed.
    - `base_model: ModelReference` (or `Optional[ModelReference]`): If your function accepts `base_model` annotated with `ModelReference`, MergeKit will pass a reference to the base model if one is used in the configuration for this merge operation.

## Class-based API Implementation

For complex merges requiring granular control, implement `MergeMethod` and `Task` classes:

### Example Implementation

```python
from mergekit.merge_methods.base import MergeMethod, ConfigParameterDef
from mergekit.common import ImmutableMap, ModelReference, WeightInfo
from mergekit.graph import Task
from typing import Any, Dict, List
import torch


class CustomDependencyTask(Task[float]):
    totally_real_parameter: str

    # Example of a task that computes a dependency for the merge
    def execute(self) -> float:
        # Custom logic to compute a dependency
        return 42.0

class CustomMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    parameters: ImmutableMap[str, Any]
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {
            "tensors": self.gather_tensors,
            "dependency": CustomDependencyTask(totally_real_parameter="example"),
        }

    def priority(self) -> int:
        return 1  # Optional: higher priority = earlier execution

    def group_label(self) -> str:
        return self.weight_info.name  # Optional: modify task grouping

    def uses_accelerator(self) -> bool:
        return True  # Enable GPU acceleration

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], dependency: float
    ) -> torch.Tensor:
        # Implementation using self.weight_info, self.parameters, and self.tensor_parameters
        # Access global parameters via self.parameters["param_name"]
        # Access per-model tensor parameters via self.tensor_parameters[model_ref]["tensor_param_name"]
        # These values are pre-resolved by MergeKit's configuration system.

        result = ...
        return result


class CustomMerge(MergeMethod):
    def name(self) -> str:
        return "custom_merge"

    def pretty_name(self) -> str:
        return "Custom Merge"

    def reference_url(self) -> str:
        return "https://example.com/custom"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef("threshold", float, required=False, default_value=0.5)
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef("weight", float, required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo, # Metadata about the weight being computed
        tensors: MergeTensorInput, # Internal Task that fetches input tensors
        parameters: ImmutableMap[str, Any], # Global parameters
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]], # Per-model parameters
        **kwargs, # Other context like base_model: Optional[ModelReference]
    ) -> Task:
        return CustomMergeTask(
            gather_tensors=tensors,
            parameters=parameters,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
        )
```

### Task Scheduling System

The class-based API provides fine-grained control over execution:

- **Priority Control**: Override `priority()` to influence execution order within groups
- **Task Grouping**: Use `group_label()` to batch similar operations
- **Resource Management**:
  - Automatic tensor lifecycle tracking
  - Memory optimization via early tensor eviction
  - Smart device placement for computation vs storage
- **Computation Graph**: Build complex flows by connecting multiple tasks

### Implementation Requirements

1. Task Class:
   - Must implement `execute()` with proper type annotations
   - Must implement `arguments()` to declare dependencies
   - Optionally override `priority()`, `group_label()`, `uses_accelerator()`

2. Method Class:
   - Must implement core methods: `name()`, `make_task()`
   - Optional methods: `pretty_name()`, `reference_url()`
   - Define parameters via `parameters()` and `tensor_parameters()`

### Registration

Add class-based methods to `STATIC_MERGE_METHODS` in `mergekit/merge_methods/registry.py`:

```python
from mergekit.merge_methods.my_module import CustomMerge

STATIC_MERGE_METHODS: List[MergeMethod] = [
    CustomMerge(),
    # other methods...
]
```

## Reference Implementations

1. **Linear Merge** (`mergekit.merge_methods.linear`):
   - Basic weighted averaging
   - Good example of class-based implementation

2. **Multi-SLERP** (`mergekit.merge_methods.multislerp`):
   - Hypersphere interpolation
   - Complex decorator usage example

3. **Task Arithmetic** (`mergekit.merge_methods.task_arithmetic`):
   - Advanced graph-based implementation
   - TIES/Magnitude pruning example


