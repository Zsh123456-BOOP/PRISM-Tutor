# Task Card 12: LLM Judge - DeepSeek V4 Pro

## 1. 任务目标

实现 DeepSeek V4 Pro judge client、judge prompt、JSON score schema 和 raw response 保存流程，用于开放式教学质量、scaffolding、answer leakage、clarity 和 student-facing appropriateness 评估。

## 2. 背景与设计约束

方案要求 judge 不能与 generator 是同一个模型；必须记录实际模型名、调用日期、temperature、top_p、max_tokens 和完整 judge prompt；保存 raw response 与 parsed scores。明文 API key 不得写入代码、配置、日志或 task card，必须通过环境变量或本机凭据来源注入。

## 3. 前置依赖

- [ ] 依赖 Task Card 10 的 generation outputs。
- [ ] 依赖 Task Card 11 的 leakage rule outputs，用于最终双重判断。
- [ ] 依赖可用 DeepSeek judge API 或同 schema 的备用 judge。
- [ ] 依赖 `DEEPSEEK_API_KEY` 或安全凭据来源，不读取项目内明文密钥。

## 4. 需要新增或修改的文件

```text
scripts/03_run_judge.py
configs/judge.yaml
prism_tutor/eval/judge_client.py
prism_tutor/eval/judge_prompts.py
prism_tutor/eval/judge_schema.py
prism_tutor/eval/judge_merge.py
outputs/judge_scores/*.jsonl
outputs/judge_scores/raw/*.jsonl
```

## 5. 具体执行步骤

- [ ] Step 1: 定义 judge JSON schema：mathematical_correctness、pedagogical_quality、scaffolding_quality、misconception_coverage、answer_leakage、clarity、student_facing_appropriateness、overall、reason。
- [ ] Step 2: 编写 judge prompt，输入 problem、student answer、gold context、candidate response，并要求只输出 JSON。
- [ ] Step 3: 实现 judge client，记录 requested_model、actual_model、api_date、temperature=0.0、top_p=1.0、max_tokens=768。
- [ ] Step 4: 保存每次请求的 prompt、candidate response、raw response、parsed score、latency 和错误。
- [ ] Step 5: 对 comparative judge 或多候选场景随机候选顺序，降低 position bias，并保存随机 seed 与展示顺序。
- [ ] Step 6: 合并 answer leakage：`final_leakage = rule_leakage or judge_leakage`，保留两个来源。

## 6. 边界情况与失败处理

- [ ] API timeout：有限 retry，仍失败则保存 failure entry，不删除样本。
- [ ] Judge 输出非 JSON：按与 agent 类似的 retry/repair 解析，但 raw response 必须保存。
- [ ] 实际模型名与配置不同：写入 metadata，并在报告中使用 actual_model。
- [ ] API 不稳定：允许切换备用 judge，但必须同 schema、同 prompt 模板，并在 manifest 标记。
- [ ] Judge 与 rule leakage 冲突：保留冲突样本，供 human audit hard sample 使用。

## 7. 验收标准

- [ ] `python scripts/03_run_judge.py --input outputs/generations --judge_config configs/judge.yaml` 可生成 judge JSONL。
- [ ] 每条 judge 记录包含 raw response 和 parsed score。
- [ ] metadata 包含 actual_model、api_date、temperature、top_p、max_tokens、prompt_version。
- [ ] answer leakage 输出 rule、judge 和 final 三个字段。
- [ ] 不会在任何输出中暴露 API key。

## 8. 不允许做的事情

- [ ] 不允许使用 Qwen3-8B 作为 judge。
- [ ] 不允许把 judge score 当作 gold label 指标替代品。
- [ ] 不允许丢弃 judge 失败样本。
- [ ] 不允许在项目文件中保存明文 API key。

## 9. 完成后产物

```text
configs/judge.yaml
scripts/03_run_judge.py
outputs/judge_scores/*.jsonl
outputs/judge_scores/raw/*.jsonl
outputs/judge_scores/judge_metadata.json
```
