# 32导脑电帽 — Motor Imagery 与 EEG Foundation Model 调研

> 面向本项目硬件：32导干电极帽，ADS1299（24bit，增益24，VREF 4.5V，LSB≈0.5364µV），
> WiFi/UDP（192.168.4.1:8086，B启动/S停止/TX打trigger）。无阻抗检测，未接M1/M2。
> 目标：尽量"少训练/免逐人校准"地做 Motor Imagery（MI）；顺带调研 EEG foundation model 用于后续数据采集测试。

---

## 0. 你的导联（从 `32导电极接口对应关系.xlsx` 提取）

32个头皮通道（去掉 REF、GND、未接的 M1/M2）：

```
FP1 FP2 AF3 AF4 F3 F4 F7 F8
FC1 FC2 FC5 FC6 C3 C4 T7 T8
CP1 CP2 CP5 CP6 P3 P4 P7 P8
PO3 PO4 O1 O2 FZ CZ PZ OZ
```

**对 MI 很关键的结论**：这套导联的**感觉运动区覆盖非常好** —— C3/Cz/C4 + 完整的 FC 环（FC1/2/5/6）+ CP 环（CP1/2/5/6）。
这基本是 BCI Competition IV-2a（22导）所需通道的超集，MI 的 μ(8–13Hz)/β(13–30Hz) ERD/ERS 主要就落在 C3/C4 附近，硬件条件是够的。

**参考问题（无 M1/M2）**：MI 不需要乳突参考。用 **CAR（共平均参考）** 或 **C3/C4 的小拉普拉斯（surrounding Laplacian）** 即可，反而对 MI 更好。需向厂商确认 REF 电极实际位置（常见是 CPz/FCz/前额）。

---

## 1. 先摆事实：MI "完全免训练" 到底可行吗？

**结论：严格意义的"零训练即插即用、可靠 MI"在一顶全新的干电极帽上不现实。** MI 是常见 BCI 范式里**跨域迁移最难**的一个：换硬件、换电极（干vs湿）、换导联/参考都会带来 domain shift。别对"下载权重直接跑"抱过高期待。

现实的能力谱系（2类"左手vs右手"想象为例，实验室湿电极的公开数据上）：

| 方案 | 是否需要目标用户数据 | LOSO/跨被试典型准确率 | 你干电极帽上的预期 |
|---|---|---|---|
| 真·零样本（完全不碰目标数据） | 否 | 不稳定，常接近偶然~60% | 不建议，作为上限参考 |
| 免校准 / 跨被试（用别人数据训练，测新人） | 否（推理时） | 2类 ~70–78%，4类 ~55–65% | 再打 5–15% 折扣 |
| **少样本校准（目标用户几分钟数据）** | 是（很少） | 明显提升，2类常 >80% | **最推荐的落地路线** |

> MOABB 大规模基准上，2类 MI 的 LOSO 平均约 **74%**（BNCI2014-001）；最强的平均解码器是 **Riemannian 切空间 + SVM** 和 **CSP** 一类方法。深度网络在加了对齐（Euclidean Alignment）后能持平或略优，但没有"碾压"。

**给本项目的定调**：把目标定为"**免逐人重训练 + 极少量校准（几分钟）**"，而不是"零数据"。并且**第一步先验证这顶帽子到底能不能测到 MI 的 ERD/ERS**（见 §6 计划），这一步 make-or-break。

---

## 2. MI 落地技术栈（按推荐顺序）

### Stack A — 经典 Riemannian（首选，最稳，无需GPU/深度训练）
- 预处理：8–30 Hz 带通 + 50Hz 陷波 + CAR；cue 后 0.5–3.5s 取 epoch；选感觉运动通道（C3,Cz,C4,FC1/2/5/6,CP1/2/5/6,可加 CP/FC 全环）。
- **pyRiemann**：协方差(OAS/shrinkage) → 切空间(Tangent Space) → LogReg/SVM（`TSclassifier`），或 `MDM`。
- 跨被试免校准：`pyriemann.transfer` 里的 **Euclidean Alignment(EA) / Riemannian 重心对齐(TLCenter)** + **MDWM**。
- 为什么先用它：MOABB 证明其为 MI 平均最强、对噪声鲁棒、可解释、训练秒级。**干电极噪声大，它的鲁棒性尤其值钱。**

### Stack B — Braindecode 深度网络（在公开数据上做被试无关训练，再迁移）
- 模型：**EEGNet**、**ShallowFBCSPNet**、**ATCNet**、**EEG-Conformer**。
- 在 **OpenBMI/Lee2019（54被试，左右手）** + PhysioNet 上做 subject-independent 训练，配 **Euclidean Alignment**。
- 部署时冻结；可选目标用户少样本微调。

### Stack C — Foundation model 探针（见 §4）
- 冻结预训练 backbone（LaBraM / EEGPT / CBraMod）+ 线性/MLP 探针。
- 仍需**少量**目标标注数据来拟合探针（分钟级），不是全量训练。

> **坦白讲**：能"下载即用、直接套到你帽子上"的 MI 权重基本不存在（导联/硬件不匹配）。可行做法是：**小分类头在公开数据或你的试点数据上快速训练/对齐**。所以 Stack A 通常是投入产出比最高的起点。

---

## 3. 开源 MI 模型 / 权重清单

| 项目 | 仓库 | 权重 | 训练数据 | 期望通道/采样率 | 跨被试 | License |
|---|---|---|---|---|---|---|
| EEGNet / ShallowConvNet / ATCNet / EEG-Conformer | **braindecode** | 提供架构，自训 | 各种 | 灵活 | 配EA可 | BSD-3 |
| 切空间/MDM + 迁移对齐 | **pyRiemann** | 无（拟合极快） | — | 灵活 | 是(EA/RA) | BSD-3 |
| Multi-branch CNN-LSTM 跨被试 | `unimib-islab/EEG-BCI-Cross-Subject-Motor-Imagery` | **提供预训练**（含64→32通道重训版本，宣称可 zero-shot 推理+短微调） | BCI IV 2a 等 | 22/32 | 是 | 待核实 |
| EEG-Conformer | `eeyhsong/EEG-Conformer` | 按数据集自训 | 2a/2b | 22/3 | 中等 | 见仓库 |
| EEGCCT（紧凑卷积Transformer，subject-independent） | 论文/Sci.Rep. 2024 | 部分开源 | 2a/2b LOSO | 22/3 | 是 | 见论文 |

**最接近"跨被试预训练权重"的候选**：`unimib-islab/EEG-BCI-Cross-Subject-Motor-Imagery`（有 64→32 通道重训模型）。**需核实其导联映射、采样率、License 是否与本帽兼容**——它的32通道未必等于你的32通道。

**基准/复现框架**：**MOABB**（一行命令跑几十条 pipeline × 多个 MI 数据集，含 LOSO），配合 **MNE-Python** 做预处理。

---

## 4. EEG Foundation Models（次要目标：采数后测试）

| 模型 | 会议 | 规模 | 预训练数据 | 权重 | 通道灵活? | License | 备注 |
|---|---|---|---|---|---|---|---|
| BENDR | 2021 | — | TUEG | GitHub | 较固定 | — | wav2vec式，偏早期 |
| BIOT | NeurIPS'23 | ~3M | 多源 | GitHub/HF | 是（通道tokenize） | — | 支持可变通道/长度 |
| **LaBraM** | ICLR'24 spotlight | base/large/huge | ~2000h | `935963004/LaBraM`（`labram-base.pth`） | **是（按电极名做channel embedding）** | 见仓库 | 最流行、强基线 |
| **EEGPT** | NeurIPS'24 | 10M | 混合 58ch/256Hz/4s | `BINE022/EEGPT`（`eegpt_..._large4E.ckpt`） | **是（channel embedding）** | 见仓库 | 论文里 **MI 线性探针优于 BENDR/BIOT/LaBraM** |
| **CBraMod** | ICLR'25 | — | 9000h TUEG(19ch) | HF `weighting666/CBraMod` | **是（criss-cross 时空注意力）** | **MIT** | 输入 (ch, seg, 200pts/patch)，即200Hz、1s/patch |
| NeuroLM | ICLR'25 | 最大1.7B | ~25000h | `935963004/NeuroLM` | 是 | 见仓库 | LLM桥接、多任务 |
| 其他 | — | — | — | — | — | — | Brant(偏iEEG)、Brain-JEPA、EEGFormer、MMM(通道无关) |

**对你32导最合适的三个**（都通道灵活、patch式、权重公开）：**EEGPT、CBraMod、LaBraM**。

**上手建议**：
1. 先试 **EEGPT**（模型小、MI 有实证、线性探针即可）与 **CBraMod**（MIT 许可、输入格式清晰）。
2. 采样率对齐：EEGPT→256Hz，CBraMod→200Hz，LaBraM→200Hz（重采样）。
3. 通道名映射：你有标准10-20命名，直接喂给它们的 channel embedding。
4. 冻结 backbone + 线性探针；用你的试点数据（分钟级）拟合探针即可评估。

**评测参考**：`EEG-FM-Bench`（2025，系统评测各 foundation model）可用来横向比较。

---

## 5. 可用公开数据集（预训练 / 探针 / 基准）

| 数据集 | MOABB 名 | 被试 | 通道/采样 | 类别 | 用途 |
|---|---|---|---|---|---|
| OpenBMI / Lee2019 | `Lee2019_MI` | **54** | 62/1000Hz | 左/右手 | **跨被试首选（量大）** |
| BCI IV 2a | `BNCI2014_001` | 9 | 22/250Hz | 4类 | 标准基准 |
| BCI IV 2b | `BNCI2014_004` | 9 | 3(双极)/250Hz | 2类 | 轻量2类 |
| PhysioNet EEGMMIDB | `PhysionetMI` | 109 | 64/160Hz | 手/脚等 | 量大但已知有标注/质量问题 |
| Cho2017 / Shin2017 / Weibo2014 / Zhou2016 | 同名 | 若干 | — | MI | 补充多样性 |

---

## 6. 针对本硬件的实操要点

- **参考**：无需 M1/M2。用 **CAR** 或 C3/C4 **拉普拉斯**。确认 REF 位置。
- **干电极**：噪声/漂移大、易受运动伪迹。8–30Hz 带通能滤掉大部分；用 **autoreject** 剔坏段/坏道。
- **无阻抗硬件 → 软件替代**：用每-epoch 方差、50Hz 线噪占比、通道间相关性做"信号质量代理指标"，在采集软件里给个红/绿提示。
- **采样率**：ADS1299 常见 250/500Hz，**需确认**；按模型重采样（256/200Hz）。
- **Trigger**：硬件支持 UDP `TX` 打标，MI cue 打标没问题——这是做有监督校准/评测的前提。
- **换算**：采样值 × 0.02235 ≈ µV（增益24）；解析注意 24bit 有符号（`&0x800000` 判负、减 16777216）。

---

## 7. 建议的初期探索路线（可执行）

1. **数据管道**：解析 UDP 数据包 → 构建 MNE `Raw`（用上面导联 + 标准10-20坐标）→ CAR → 8–30Hz。
2. **采一段 MI 试点**：左手 vs 右手想象，各~40 trial，带 trigger。即使目标是"少训练"，也需要它来 (a) 验证信号质量、(b) 拟合轻量探针/对齐参数。
3. **关键 sanity check**：画 ERD/ERS 地形图 + C3/C4 μ 功率 左vs右 对比。**能看到对侧 μ/β 功率下降 = 这顶帽子能做 MI**；看不到就先解决硬件/贴合问题，别急着上模型。
4. **基线**：pyRiemann 切空间+LR，先 within-subject，再跨被试(EA)，用 MOABB 风格评测。
5. **Foundation 探针**：EEGPT/CBraMod 冻结 + 线性探针，在试点数据上对比 Stack A。
6. **对比迭代**：选准确率/稳定性最好的组合。

---

## 主要来源
- EEGPT (NeurIPS 2024) — https://github.com/BINE022/EEGPT ; https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
- LaBraM (ICLR 2024) — https://github.com/935963004/LaBraM
- CBraMod (ICLR 2025) — https://github.com/wjq-learning/CBraMod ; 权重 https://huggingface.co/weighting666/CBraMod
- NeuroLM (ICLR 2025) — https://github.com/935963004/NeuroLM
- 跨被试 MI 预训练 — https://github.com/unimib-islab/EEG-BCI-Cross-Subject-Motor-Imagery
- Euclidean Alignment 综述 — https://arxiv.org/pdf/2502.09203
- Riemannian 迁移/免校准 — https://arxiv.org/pdf/2111.12071
- MOABB 基准 — https://github.com/NeuroTechX/moabb
- pyRiemann — https://pyriemann.readthedocs.io
- Braindecode — https://braindecode.org
- EEG-FM-Bench (2025) — https://arxiv.org/pdf/2508.17742
- EEG foundation models 汇总 — https://github.com/gayalkuruppu/eeg-foundation-models
