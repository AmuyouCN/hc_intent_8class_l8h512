# hc_intent_8class_l8h512 Evaluation Report

## Evaluation Dataset

Current public metrics are reported on `data/eval_standard.csv`.

This dataset contains 800 Chinese intent-classification samples, with 100 samples for each of the 8 labels. It was not used for model training, hyperparameter tuning, or model selection.

## Overall Metrics

| Metric | Value |
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
| Avg Confidence | 81.69% |

## Per-Class Metrics

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 服务咨询 | 89.47% | 85.00% | 87.18% | 100 |
| 服务办理 | 93.07% | 94.00% | 93.53% | 100 |
| 服务变更 | 93.14% | 95.00% | 94.06% | 100 |
| 护理记录 | 87.50% | 84.00% | 85.71% | 100 |
| 健康咨询 | 72.66% | 93.00% | 81.58% | 100 |
| 紧急处理 | 86.09% | 99.00% | 92.09% | 100 |
| 人工联系 | 90.91% | 80.00% | 85.11% | 100 |
| 投诉反馈 | 96.00% | 72.00% | 82.29% | 100 |

## Notes

- The model keeps high recall on emergency handling, but this does not make it suitable for medical decision-making.
- Complaint feedback and human-contact requests remain weaker recall areas.
- Short or ambiguous inputs should be paired with human review or fallback handling in production workflows.
