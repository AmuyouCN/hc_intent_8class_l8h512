# hc_intent_8class_l8h512

面向居家护理 / 健康护理场景的中文 8 类意图识别模型。

## 上游模型与许可

本模型基于 `uer/chinese_roberta_L-8_H-512` 微调得到。该上游模型属于 UER Chinese RoBERTa Miniatures 系列，相关项目 UER-py 采用 Apache License 2.0。

本项目保留对上游模型和 UER-py 项目的署名说明。仓库中的自有代码、文档和脚本按本仓库 LICENSE 授权发布；模型权重包含基于上游模型微调得到的参数，使用、修改或分发时请同时遵守上游项目的许可要求。

## 免责声明

本模型仅用于意图分类与技术研究，不构成医疗建议、诊断或治疗意见。实际护理、医疗与应急决策请以专业人员判断为准。

## 数据说明

训练与评估数据均不包含真实患者隐私数据。公开评估结果基于独立标准评估集 `data/eval_standard.csv`；该评估集未用于模型训练、调参或模型选择。

## 标签设计说明

本模型的 8 个意图标签围绕居家护理场景中的常见用户请求整理，结合护理服务流程、居家医养结合和患者安全等公开资料细化而成。相关资料主要用于标签设计参考，不代表某个单一标准文件的固定分类。

这 8 个意图按“服务生命周期 + 风险优先级”设计，既覆盖用户从了解服务、办理服务、调整服务到查看服务结果的主要路径，也单独区分健康安全、人工介入和服务质量反馈等关键场景。

| 分组 | 意图 | 设计目的 |
|---|---|---|
| 服务流程类 | 服务咨询 / 服务办理 / 服务变更 / 护理记录 | 覆盖咨询前、办理中、服务调整和服务后的主要流程节点 |
| 健康安全类 | 健康咨询 / 紧急处理 | 将一般健康咨询与高风险紧急场景分开，降低把危险情况误当普通咨询的风险 |
| 人工协同类 | 人工联系 | 识别需要人工客服、护士或护理员介入的请求，便于触发人工流程 |
| 质量反馈类 | 投诉反馈 | 保留投诉、建议和服务质量反馈入口，便于后续工单和质控处理 |

主要参考资料：

- [国家卫生健康委：居家和社区医养结合服务指南（试行）](https://www.nhc.gov.cn/lljks/c100158/202311/7c3432a4cf1243ccbd03d31d54879bcc.shtml)
- [国家卫生健康委：进一步改善护理服务行动计划（2023-2025年）](https://www.nhc.gov.cn/yzygj/c100068/202306/8fe28be0f8e241cb8444b4f242706495.shtml)
- [国家卫生健康委：关于开展老年护理需求评估和规范服务工作的通知](https://www.nhc.gov.cn/yzygj/c100068/201908/01a2813f59bc44a0aaab11f0f9fda3f7.shtml)
- [国家卫生健康委：《医疗机构投诉管理办法》解读](https://www.nhc.gov.cn/fzs/c100047/201903/da7b9057f37744fc9745bf1e736bbbe9.shtml)
- [WHO: Patient safety](https://www.who.int/news-room/fact-sheets/detail/patient-safety)
- [WHO: Global Patient Safety Action Plan 2021-2030](https://www.who.int/teams/integrated-health-services/patient-safety/policy/global-patient-safety-action-plan)

## 标签体系

| ID | 意图 |
|---:|---|
| 0 | 服务咨询 |
| 1 | 服务办理 |
| 2 | 服务变更 |
| 3 | 护理记录 |
| 4 | 健康咨询 |
| 5 | 紧急处理 |
| 6 | 人工联系 |
| 7 | 投诉反馈 |

## 评估结果

### Standard Independent Evaluation

| 指标 | 值 |
|---|---:|
| Samples | 800 |
| Accuracy | 87.75% |
| Macro F1 | 87.69% |
| Macro Precision | 88.60% |
| Macro Recall | 87.75% |
| Min Recall | 72.00% |
| Bottom3 Avg | 78.67% |
| Emergency Recall | 99.00% |
| Non-emergency to Emergency FP Rate | 2.29% |

### Per-class metrics

| 意图 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 服务咨询 | 89.47% | 85.00% | 87.18% |
| 服务办理 | 93.07% | 94.00% | 93.53% |
| 服务变更 | 93.14% | 95.00% | 94.06% |
| 护理记录 | 87.50% | 84.00% | 85.71% |
| 健康咨询 | 72.66% | 93.00% | 81.58% |
| 紧急处理 | 86.09% | 99.00% | 92.09% |
| 人工联系 | 90.91% | 80.00% | 85.11% |
| 投诉反馈 | 96.00% | 72.00% | 82.29% |

## 使用方式

### 1. 安装依赖

建议先在项目目录中创建并激活虚拟环境，再安装依赖，避免把依赖安装到系统 Python 环境中。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. 下载模型权重

本模型权重可通过以下平台获取：

| 平台 | 地址 |
|---|---|
| GitHub Releases | [https://github.com/AmuyouCN/hc_intent_8class_l8h512/releases/latest](https://github.com/AmuyouCN/hc_intent_8class_l8h512/releases/latest) |
| Hugging Face | [https://huggingface.co/amuyoucn/hc_intent_8class_l8h512](https://huggingface.co/amuyoucn/hc_intent_8class_l8h512) |
| ModelScope 魔搭社区 | [https://modelscope.cn/models/AmuyouCN/hc_intent_8class_l8h512](https://modelscope.cn/models/AmuyouCN/hc_intent_8class_l8h512) |

### 3. 单条预测

```powershell
python scripts/predict.py "老人发烧了怎么办"
```

示例输出：

```text
Text: 老人发烧了怎么办
  Intent: 5 (紧急处理), Confidence: 0.9487
```

### 4. 批量预测

准备一个文本文件，每行一条文本，例如 `examples.txt`：

```text
老人发烧了怎么办
我要预约上门护理
帮我联系人工客服
```

运行：

```powershell
python scripts/predict.py --file examples.txt
```

### 5. 运行评估

```powershell
python scripts/evaluate.py
```

评估结果会写入：

```text
eval_results/
```

### 6. 启动 FastAPI 服务

```powershell
python scripts/serve_fastapi.py --host 127.0.0.1 --port 8001 --device auto
```

### 7. HTTP 调用示例

```powershell
curl -X POST http://127.0.0.1:8001/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"老人发烧了怎么办\"}"
```

## 微调说明

本仓库主要提供已训练模型、推理脚本和评估脚本，不包含完整训练流水线。

如需在自己的数据上继续微调，建议保持以下数据格式：

```csv
text,intent_id,intent_name
老人发烧了怎么办,5,紧急处理
我要预约上门护理,1,服务办理
```

微调时请保持标签体系不变：

```text
0 服务咨询
1 服务办理
2 服务变更
3 护理记录
4 健康咨询
5 紧急处理
6 人工联系
7 投诉反馈
```

微调完成后，需要替换：

```text
model/model.pt
```

如果修改了模型结构或标签映射，还需要同步更新：

```text
model/config.json
model/label_map.json
```

建议微调后重新运行：

```powershell
python scripts/evaluate.py
```

如果更新了 `model/` 下的核心文件，请重新生成 `SHA256SUMS`。

## FAQ / 已知问题

**这个模型能提供医疗建议吗？**  
不能。它只做意图分类，不提供诊断、治疗或护理决策。

**是否必须使用 GPU？**  
不必须。脚本会自动使用可用 GPU，否则使用 CPU。

**为什么有些边界表达可能不稳定？**  
短文本或上下文不足时，部分意图天然存在歧义，建议在业务系统中保留人工复核入口。

## 引用

如需引用本模型，请参见 `CITATION.cff`。

## 文件结构

```text
model/
  model.pt
  config.json
  label_map.json
data/
  eval_standard.csv
scripts/
  predict.py
  evaluate.py
  serve_fastapi.py
eval_results/
  standard/
docs/
  model_card.md
  eval_report.md
CITATION.cff
SHA256SUMS
```
