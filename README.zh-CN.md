<div align="center">

# rlm-saga-control

**在 LLM 场景推理上加一层事务性 Saga 控制,被打断的计划能否被干净地恢复**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

[한국어](./README.md) · [English](./README.md#english) · **中文**

</div>

> 🧊 **休眠中的研究试点。**

## ⭐ 核心结果 (TL;DR)

- **完整的 Saga 式控制明显胜过轻量分支**——在未见过的场景上准确率显著更高(最后一次完整的配对比较)。
- 但**多项自定的运营指标未达标**(延迟、闸门误报率、fallback 调用频率)——"能用,但还没打磨好"。
- 下一轮的代码与配置已就绪,但需要更大的环境(多 GPU + 独立服务化),故暂停。

## 这项研究想看什么

让语言模型去解决一个计划/日程类的场景(比如安排一天的行程),如果中途突然冒出一个意料之外的事件(disruption / 教란),整个计划往往就乱了。本项目的问题是:**在 LLM 之上加一层从事务系统借来的安全网(Saga 风格的控制层)**,能不能只干净地修复被打乱的那一段,而把其余决定保留下来?

主要比较了三种作法:

- 放任不管 —— 让模型自己处理。
- 只做轻量分支 —— 把扰动那段切出来,前后单独处理。
- 更完整的控制 —— **prefix lock(前缀锁定)+ 校验 + 恢复 + 不变性管理**一起上。

评估不只看简单的成功率,还看运营指标:扰动前的决定有没有被弄坏、修复是否一致、增加了多少延迟、fallback 调用频率等。

## 发现了什么

- **更完整的控制这一侧明显更好。** 在最后一次完整跑完的比较里(在未见过的场景上做的 paired 比较),它比"只做分支"那侧的准确率有意义地更高。
- **但运营指标的自定目标线没有都达成。** 准确率上去了,但延迟、误通过 (false positive) 率、fallback 调用频率这些项,有几个没过自己设的门槛。
- **也就是"能用但还没打磨好"的状态。** 下一轮要打磨的代码与配置已经准备好了,但实际执行没跑到完结就停了。

完整数字可以在以下两份关闭报告中查阅:

- 🇰🇷 [`closure_reports/project_closure_report_ko_20260327.md`](closure_reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`closure_reports/project_closure_report_20260327.md`](closure_reports/project_closure_report_20260327.md)

## 为什么暂停

正向信号本身是清楚的,下一步要做什么也已经整理好。但要说"达到可投入运营级别",还需要再打磨一轮,而那一轮所需要的环境(多 GPU、单独的服务基础设施)比现在更大。因此现在先把它暂停下来,等下一轮可以做的时候再唤醒,是比较自然的选择。

## 重启时先看哪里

- 📖 [`GLOSSARY.md`](GLOSSARY.md) —— 把代码 row、关闭报告、分析表里出现的内部术语(三种 regime、gate / gate FP、不变性、prefix lock、fallback、strict vs loose、`SAGA_*` 环境变量)翻成日常用语的对照表
- [`PROJECT_HANDOVER_KO.md`](PROJECT_HANDOVER_KO.md) —— 交接文档（仅韩文），建议第一个看
- [`code/exp/README.md`](code/exp/README.md) —— 代码树入口
- [`evidence/`](evidence/) —— 最后一次完整 paired 比较的代表性产出(paired 比较、gate 对齐报告、失败分解)

重新启动时建议的优先级:**提升 gate 精度 → 强化确定性恢复逻辑 → 降低 fallback 频率 → 削减延迟。**

## 代码地图

| 文件 | 做什么 |
|---|---|
| [`code/exp/runners/common.py`](code/exp/runners/common.py) | 实验循环的共享部件:gate、retry、recovery |
| [`code/exp/runners/run_paired.py`](code/exp/runners/run_paired.py) | paired 比较实验的 runner |
| [`code/exp/analysis/gate_fp_breakdown.py`](code/exp/analysis/gate_fp_breakdown.py) | "gate 通过但 strict 失败"案例的原因分解 |
| [`code/exp/analysis/state_timeline_failure_breakdown.py`](code/exp/analysis/state_timeline_failure_breakdown.py) | 逐时间点的 state 一致性违规分解 |
| [`code/exp/analysis/immutability_alignment_report.py`](code/exp/analysis/immutability_alignment_report.py) | 检查不应改变的部分是否被良好保留 |
| [`code/exp/bench/scorer.py`](code/exp/bench/scorer.py) | 结果打分(loose / strict 两套标准) |

## 目录概览

```
.
├── code/exp/              实验代码(runners、analysis、bench、config)
├── evidence/              最后一次完整比较的代表性产出
├── manifests/             包含 run 列表与校验和
├── closure_reports/       关闭报告(韩文 / 英文)
├── GLOSSARY.md            内部术语词典
└── PROJECT_HANDOVER_KO.md 交接文档
```

代码中嵌入了绝对路径(`/disk/...`)假设和多 GPU + 单独服务环境的前提,直接拿到新环境上是跑不起来的。归档更适合用作"重新理解并唤醒"的起点,而不是直接重跑。

## 环境

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# 需要单独的服务环境(例如 vLLM,多 GPU)
```

bootstrap 脚本和 Python runner 对原作者本机绝对路径默认值都提供了环境变量覆盖。`run_*_with_cleanup.sh` 这一系列 wrapper 脚本作为历史运行记录原样保留,没有被参数化;如果要尝试重新执行,实际入口是 `bootstrap_env.sh`。

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `SAGA_EXPERIMENTS_ROOT` | run 目录创建的位置 | `/disk/chojm/experiments` |
| `SAGA_LLM_REQUIREMENTS` | bootstrap 时安装的 requirements.txt | `/disk/chojm/SagaLLM/requirements.txt` |
| `SAGA_RLM_PACKAGE` | 以 editable 模式安装的 RLM 包路径 | `/disk/chojm/rlm` |
| `RLM_SAGA_RUN_ROOT_BASE` | Python runner 的 `run_root` 默认值(优先于 YAML) | YAML 中的 `paths.run_root_base` |

`rlm` 包**未随本归档捆绑**。`SAGA_RLM_PACKAGE` 由 `bootstrap_env.sh` 用于 `pip install -e` 安装；Python runner 同时也期望该包位于 `<workspace>/rlm/` 并已加入 `sys.path`。在执行 bootstrap 前，请将 `SAGA_RLM_PACKAGE` 设置为本地 RLM 代码仓库的路径。

## 状态

🧊 **休眠中** —— 明确的正向信号和准备好的下一步,在那一轮真正开跑之前暂停。

## 许可证

以 [CC BY-NC 4.0](./LICENSE) 发布。
