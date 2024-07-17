# 通用信息抽取 UIE(Universal Information Extraction) PyTorch版

**迁移[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)中的UIE模型到PyTorch上**

* 2022-10-3: 新增对UIE-M系列模型的支持，增加了ErnieM的Tokenizer。ErnieMTokenizer使用C++实现的高性能分词算子FasterTokenizer进行文本预处理加速。需要通过`pip install faster_tokenizer`安装FasterTokenizer库后方可使用。

PyTorch版功能介绍
- `convert.py`: 自动下载并转换模型，详见[开箱即用](#开箱即用)。
- `doccano.py`: 转换标注数据，详见[数据标注](#数据标注)。
- `evaluate.py`: 评估模型，详见[模型评估](#模型评估)。
- `export_model.py`: 导出ONNX推理模型，详见[模型部署](#模型部署)。
- `finetune.py`: 微调训练，详见[模型微调](#模型微调)。
- `model.py`: 模型定义。
- `uie_predictor.py`: 推理类。


## 1. 模型简介

[UIE(Universal Information Extraction)](https://arxiv.org/pdf/2203.12277.pdf)：Yaojie Lu等人在ACL-2022中提出了通用信息抽取统一框架UIE。该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。为了方便大家使用UIE的强大能力，PaddleNLP借鉴该论文的方法，基于ERNIE 3.0知识增强预训练模型，训练并开源了首个中文通用信息抽取模型UIE。该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167236006-66ed845d-21b8-4647-908b-e1c6e7613eb1.png height=400 hspace='10'/>
</div>

#### UIE的优势

- **使用简单**：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**。

- **降本增效**：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，**大幅度降低标注数据依赖，在降低成本的同时，还提升了效果**。

- **效果领先**：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。

<a name="应用示例"></a>


## 2. 开箱即用

```uie_predictor```提供通用信息抽取、评价观点抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）、以及评价维度、观点词、情感倾向等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

```uie_predictor```现在可以自动下载模型了，**无需手动convert**，如果想手动转换模型，可以参照以下方法。

**下载并转换模型**，将下载Paddle版的`uie-base`模型到当前目录中，并生成PyTorch版模型`uie_base_pytorch`。

```shell
python convert.py
```

如果没有安装paddlenlp，则使用以下命令。这将不会导入paddlenlp，以及不会验证转换结果正确性。

```shell
python convert.py --no_validate_output
```

可配置参数说明：

- `input_model`: 输入的模型所在的文件夹，例如存在模型`./model_path/model_state.pdparams`，则传入`./model_path`。如果传入`uie-base`或`uie-tiny`等在模型列表中的模型，且当前目录不存在此文件夹时，将自动下载模型。默认值为`uie-base`。
  
  支持自动下载的模型
  - `uie-base`
  - `uie-medium`
  - `uie-mini`
  - `uie-micro`
  - `uie-nano`
  - `uie-medical-base`
  - `uie-tiny` (弃用，已改为`uie-medium`)
  - `uie-base-en`
  - `uie-m-base`
  - `uie-m-large`
  - `ernie-3.0-base-zh`*

- `output_model`: 输出的模型的文件夹，默认为`uie_base_pytorch`。
- `no_validate_output`：是否关闭对输出模型的验证，默认打开。

\* : 使用`ernie-3.0-base-zh`时不会验证模型，需要微调后才能用于预测


<a name="实体抽取"></a>

#### 2.1 实体抽取

  命名实体识别（Named Entity Recognition，简称NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，抽取的类别没有限制，用户可以自己定义。

  - 例如抽取的目标实体类型是"时间"、"选手"和"赛事名称", schema构造如下：

  ```text
  ['时间', '选手', '赛事名称']
  ```

    调用示例：

  ```python
  >>> from uie_predictor import UIEPredictor
  >>> from pprint import pprint

  >>> schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
  >>> ie = UIEPredictor(model='uie-base', schema=schema)
  >>> pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
  [{'时间': [{'end': 6,
            'probability': 0.9857378532924486,
            'start': 0,
            'text': '2月8日上午'}],
    '赛事名称': [{'end': 23,
              'probability': 0.8503089953268272,
              'start': 6,
              'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
    '选手': [{'end': 31,
            'probability': 0.8981548639781138,
            'start': 28,
            'text': '谷爱凌'}]}]
  ```

  - 例如抽取的目标实体类型是"肿瘤的大小"、"肿瘤的个数"、"肝癌级别"和"脉管内癌栓分级", schema构造如下：

  ```text
  ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
  ```

  在上例中我们已经实例化了一个`UIEPredictor`对象，这里可以通过`set_schema`方法重置抽取目标。

    调用示例：

  ```python
  >>> schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
  >>> ie.set_schema(schema)
  >>> pprint(ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"))
  [{'肝癌级别': [{'end': 20,
              'probability': 0.9243267447402701,
              'start': 13,
              'text': 'II-III级'}],
    '肿瘤的个数': [{'end': 84,
              'probability': 0.7538413804059623,
              'start': 82,
              'text': '1个'}],
    '肿瘤的大小': [{'end': 100,
              'probability': 0.8341128043459491,
              'start': 87,
              'text': '4.2×4.0×2.8cm'}],
    '脉管内癌栓分级': [{'end': 70,
                'probability': 0.9083292325934664,
                'start': 67,
                'text': 'M0级'}]}]
  ```

  - 例如抽取的目标实体类型是"person"和"organization"，schema构造如下：

    ```text
    ['person', 'organization']
    ```

    英文模型调用示例：

    ```python
    >>> from uie_predictor import UIEPredictor
    >>> from pprint import pprint
    >>> schema = ['Person', 'Organization']
    >>> ie_en = UIEPredictor(model='uie-base-en', schema=schema)
    >>> pprint(ie_en('In 1997, Steve was excited to become the CEO of Apple.'))
    [{'Organization': [{'end': 53,
                        'probability': 0.9985840259877357,
                        'start': 48,
                        'text': 'Apple'}],
      'Person': [{'end': 14,
                  'probability': 0.999631971804547,
                  'start': 9,
                  'text': 'Steve'}]}]
    ```


<a name="模型选择"></a>

#### 2.2 模型选择

- 多模型选择，满足精度、速度要求

  | 模型 |  结构  | 语言 |
  | :---: | :--------: | :--------: |
  | `uie-base` (默认)| 12-layers, 768-hidden, 12-heads | 中文 |
  | `uie-base-en` | 12-layers, 768-hidden, 12-heads | 英文 |
  | `uie-medical-base` | 12-layers, 768-hidden, 12-heads | 中文 |
  | `uie-medium`| 6-layers, 768-hidden, 12-heads | 中文 |
  | `uie-mini`| 6-layers, 384-hidden, 12-heads | 中文 |
  | `uie-micro`| 4-layers, 384-hidden, 12-heads | 中文 |
  | `uie-nano`| 4-layers, 312-hidden, 12-heads | 中文 |
  | `uie-m-large`| 24-layers, 1024-hidden, 16-heads | 中、英文 |
  | `uie-m-base`| 12-layers, 768-hidden, 12-heads | 中、英文 |


- `uie-nano`调用示例：

  ```python
  >>> from uie_predictor import UIEPredictor

  >>> schema = ['时间', '选手', '赛事名称']
  >>> ie = UIEPredictor('uie-nano', schema=schema)
  >>> ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
  [{'时间': [{'text': '2月8日上午', 'start': 0, 'end': 6, 'probability': 0.6513581678349247}], '选手': [{'text': '谷爱凌', 'start': 28, 'end': 31, 'probability': 0.9819330659468051}], '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 23, 'probability': 0.4908131110420939}]}]
  ```

- `uie-m-base`和`uie-m-large`支持中英文混合抽取，调用示例：

  ```python
  >>> from pprint import pprint
  >>> from uie_predictor import UIEPredictor

  >>> schema = ['Time', 'Player', 'Competition', 'Score']
  >>> ie = UIEPredictor(schema=schema, model="uie-m-base", schema_lang="en")
  >>> pprint(ie(["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！", "Rafael Nadal wins French Open Final!"]))
  [{'Competition': [{'end': 23,
                    'probability': 0.9373889907291257,
                    'start': 6,
                    'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
    'Player': [{'end': 31,
                'probability': 0.6981119555336441,
                'start': 28,
                'text': '谷爱凌'}],
    'Score': [{'end': 39,
              'probability': 0.9888507878270296,
              'start': 32,
              'text': '188.25分'}],
    'Time': [{'end': 6,
              'probability': 0.9784080036931151,
              'start': 0,
              'text': '2月8日上午'}]},
  {'Competition': [{'end': 35,
                    'probability': 0.9851549932171295,
                    'start': 18,
                    'text': 'French Open Final'}],
    'Player': [{'end': 12,
                'probability': 0.9379371275888104,
                'start': 0,
                'text': 'Rafael Nadal'}]}]
  ```

<a name="更多配置"></a>

#### 2.3 更多配置

```python
>>> from uie_predictor import UIEPredictor

>>> ie = UIEPredictor('uie_nano',   
                       schema=schema)  
```

* `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`和`uie-medical-base`, `uie-base-en`。
* `schema`：定义任务抽取目标，可参考开箱即用中不同任务的调用示例进行配置。
* `schema_lang`：设置schema的语言，默认为`zh`, 可选有`zh`和`en`。因为中英schema的构造有所不同，因此需要指定schema的语言。该参数只对`uie-m-base`和`uie-m-large`模型有效。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `task_path`：设定自定义的模型。
* `position_prob`：模型对于span的起始位置/终止位置的结果概率在0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
* `use_fp16`：是否使用`fp16`进行加速，默认关闭。`fp16`推理速度更快。如果选择`fp16`，请先确保机器正确安装NVIDIA相关驱动和基础软件，**确保CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保GPU设备的CUDA计算能力（CUDA Compute Capability）大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。

<a name="训练定制"></a>

## 3. 训练定制

对于简单的抽取目标可以直接使用```UIEPredictor```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过`报销工单信息抽取`的例子展示如何通过5条训练数据进行UIE模型微调。
<a name="代码结构"></a>

#### 3.1 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
├── finetune.py       # 模型微调脚本
├── evaluate.py       # 模型评估脚本
└── README.md
```

<a name="数据标注"></a>

#### 3.2 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考[doccano数据标注指南](doccano.md)。

原始数据示例：

```text
深大到双龙28块钱4月24号交通费
```

抽取的目标(schema)为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注步骤如下：

- 在doccano平台上，创建一个类型为``序列标注``的标注项目。
- 定义实体标签类别，上例中需要定义的实体标签有``出发地``、``目的地``、``费用``和``时间``。
- 使用以上定义的标签开始标注数据，下面展示了一个doccano标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
</div>

- 标注完成后，在doccano平台上导出文件，并将其重命名为``doccano_ext.json``后，放入``./data``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.2 0
```


可配置参数说明：

- ``doccano_file``: 从doccano导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向"]。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度级分类任务有效。默认为"##"。

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从doccano导出的文件，默认文件中的每条数据都是经过人工正确标注的。

更多**不同类型任务（关系抽取、事件抽取、评价观点抽取等）的标注规则及参数说明**，请参考[doccano数据标注指南](doccano.md)。

此外，也可以通过数据标注平台 [Label Studio](https://labelstud.io/) 进行数据标注。本示例提供了 [labelstudio2doccano.py](./labelstudio2doccano.py) 脚本，将 label studio 导出的 JSON 数据文件格式转换成 doccano 导出的数据文件格式，后续的数据转换与模型微调等操作不变。

```shell
python labelstudio2doccano.py --labelstudio_file label-studio.json
```

可配置参数说明：

- ``labelstudio_file``: label studio 的导出文件路径（仅支持 JSON 格式）。
- ``doccano_file``: doccano 格式的数据文件保存路径，默认为 "doccano_ext.jsonl"。
- ``task_type``: 任务类型，可选有抽取（"ext"）和分类（"cls"）两种类型的任务，默认为 "ext"。

<a name="模型微调"></a>

#### 3.3 模型微调

通过运行以下命令进行模型微调：

```shell
python finetune.py \
    --train_path "./data/train.txt" \
    --dev_path "./data/dev.txt" \
    --save_dir "./checkpoint" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie_base_pytorch" \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device "gpu"
```

可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为100。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，默认为`uie_base_pytorch`。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `valid_steps`: evaluate的间隔steps数，默认100。
- `device`: 选用什么设备进行训练，可选cpu或gpu。
- `max_model_num`: 保存的模型的个数，不包含`model_best`和`early_stopping`保存的模型，默认为5。
- `early_stopping`: 是否采用提前停止（Early Stopping），默认不使用。

<a name="模型评估"></a>

#### 3.4 模型评估

通过运行以下命令进行模型评估：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512
```

评估方式说明：采用单阶段评价的方式，即关系抽取、事件抽取等需要分阶段预测的任务对每一阶段的预测结果进行分别评价。验证/测试集默认会利用同一层级的所有标签来构造出全部负例。

可开启`debug`模式对每个正例类别分别进行评估，该模式仅用于模型调试：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --debug
```

输出打印示例：

```text
[2022-09-14 03:13:58,877] [    INFO] - -----------------------------
[2022-09-14 03:13:58,877] [    INFO] - Class Name: 疾病
[2022-09-14 03:13:58,877] [    INFO] - Evaluation Precision: 0.89744 | Recall: 0.83333 | F1: 0.86420
[2022-09-14 03:13:59,145] [    INFO] - -----------------------------
[2022-09-14 03:13:59,145] [    INFO] - Class Name: 手术治疗
[2022-09-14 03:13:59,145] [    INFO] - Evaluation Precision: 0.90000 | Recall: 0.85714 | F1: 0.87805
[2022-09-14 03:13:59,439] [    INFO] - -----------------------------
[2022-09-14 03:13:59,440] [    INFO] - Class Name: 检查
[2022-09-14 03:13:59,440] [    INFO] - Evaluation Precision: 0.77778 | Recall: 0.56757 | F1: 0.65625
[2022-09-14 03:13:59,708] [    INFO] - -----------------------------
[2022-09-14 03:13:59,709] [    INFO] - Class Name: X的手术治疗
[2022-09-14 03:13:59,709] [    INFO] - Evaluation Precision: 0.90000 | Recall: 0.85714 | F1: 0.87805
[2022-09-14 03:13:59,893] [    INFO] - -----------------------------
[2022-09-14 03:13:59,893] [    INFO] - Class Name: X的实验室检查
[2022-09-14 03:13:59,894] [    INFO] - Evaluation Precision: 0.71429 | Recall: 0.55556 | F1: 0.62500
[2022-09-14 03:14:00,057] [    INFO] - -----------------------------
[2022-09-14 03:14:00,058] [    INFO] - Class Name: X的影像学检查
[2022-09-14 03:14:00,058] [    INFO] - Evaluation Precision: 0.69231 | Recall: 0.45000 | F1: 0.54545
```

可配置参数说明：

- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`pytorch_model.bin`及配置文件`config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `device`: 选用进行训练的设备，可选`cpu`或`gpu`。

<a name="定制模型一键预测"></a>

#### 3.5 定制模型一键预测

`UIEPredictor`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`pytorch_model.bin`。

```python
>>> from pprint import pprint
>>> from uie_predictor import UIEPredictor

>>> schema = ['出发地', '目的地', '费用', '时间']
# 设定抽取目标和定制化模型权重路径
>>> my_ie = UIEPredictor(model='uie-base',task_path='./checkpoint/model_best', schema=schema)
>>> pprint(my_ie("城市内交通费7月5日金额114广州至佛山"))
[{'出发地': [{'end': 17,
           'probability': 0.9975287467835301,
           'start': 15,
           'text': '广州'}],
  '时间': [{'end': 10,
          'probability': 0.9999476678061399,
          'start': 6,
          'text': '7月5日'}],
  '目的地': [{'end': 20,
           'probability': 0.9998511131226735,
           'start': 18,
           'text': '佛山'}],
  '费用': [{'end': 15,
          'probability': 0.9994474579292856,
          'start': 12,
          'text': '114'}]}]
```

<a name="实验指标"></a>

#### 3.6 实验指标

我们在互联网、医疗、金融三大垂类自建测试集上进行了实验：

<table>
<tr><th row_span='2'><th colspan='2'>金融<th colspan='2'>医疗<th colspan='2'>互联网
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-base (12L768H)<td>46.43<td>70.92<td><b>71.83</b><td>85.72<td>78.33<td>81.86
<tr><td>uie-medium (6L768H)<td>41.11<td>64.53<td>65.40<td>75.72<td>78.32<td>79.68
<tr><td>uie-mini (6L384H)<td>37.04<td>64.65<td>60.50<td>78.36<td>72.09<td>76.38
<tr><td>uie-micro (4L384H)<td>37.53<td>62.11<td>57.04<td>75.92<td>66.00<td>70.22
<tr><td>uie-nano (4L312H)<td>38.94<td>66.83<td>48.29<td>76.74<td>62.86<td>72.35
<tr><td>uie-m-large (24L1024H)<td><b>49.35</b><td><b>74.55</b><td>70.50<td><b>92.66</b><td><b>78.49</b><td><b>83.02</b>
<tr><td>uie-m-base (12L768H)<td>38.46<td>74.31<td>63.37<td>87.32<td>76.27<td>80.13
</table>

0-shot表示无训练数据直接通过```UIEPredictor```进行预测，5-shot表示每个类别包含5条标注数据进行模型微调。**实验表明UIE在垂类场景可以通过少量数据（few-shot）进一步提升效果**。

<a name="模型部署"></a>

#### 3.7 模型部署

以下是UIE Python端的部署流程，包括环境准备、模型导出和使用示例。

- 环境准备
  UIE的部署分为CPU和GPU两种情况，请根据你的部署环境安装对应的依赖。

  - CPU端

    CPU端的部署请使用如下命令安装所需依赖

    ```shell
    pip install onnx onnxruntime
    ```
  - GPU端

    为了在GPU上获得最佳的推理性能和稳定性，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保**CUDA >= 11.2，cuDNN >= 8.1.1**，并使用以下命令安装所需依赖

    ```shell
    pip install onnx onnxconverter_common onnxruntime-gpu
    ```

    如需使用半精度（FP16）部署，请确保GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。
    更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)


- 模型导出

  将训练后的动态图参数导出为静态图参数：

  ```shell
  python export_model.py --model_path ./checkpoint/model_best --output_path ./export
  ```

  可配置参数说明：

  - `model_path`: 动态图训练保存的参数路径，路径下包含模型参数文件`pytorch_model.bin`和模型配置文件`config.json`。
  - `output_path`: 静态图参数导出路径，默认导出路径为`model_path`，即保存到输入模型同目录下。

- 推理

  - CPU端推理样例

    在CPU端，请使用如下命令进行部署

    ```shell
    python uie_predictor.py --task_path ./export --engine onnx --device cpu
    ```

    可配置参数说明：
    - `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`和`uie-medical-base`, `uie-base-en`。
    - `task_path`: 用于推理的ONNX模型文件所在文件夹。例如模型文件路径为`./export/inference.onnx`，则传入`./export`。如果不设置，将自动下载`model`对应的模型。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
    - `engine`: 可选值为`pytorch`和`onnx`。推理使用的推理引擎。

  - GPU端推理样例

    在GPU端，请使用如下命令进行部署

    ```shell
    python uie_predictor.py --task_path ./export --engine onnx --device gpu --use_fp16
    ```

    可配置参数说明：
    - `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`和`uie-medical-base`, `uie-base-en`。
    - `task_path`: 用于推理的ONNX模型文件所在文件夹。例如模型文件路径为`./export/inference.onnx`，则传入`./export/inference`。如果不设置，将自动下载`model`对应的模型。
    - `use_fp16`: 是否使用FP16进行加速，默认关闭。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
    - `engine`: 可选值为`pytorch`和`onnx`。推理使用的推理引擎。
