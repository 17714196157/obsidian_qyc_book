---
title: 【训练监控】WandB 与 TensorBoard 选型 + 必须记录的六大指标
source: https://www.bilibili.com/video/BV1gobj6gE67/
bvid: BV1gobj6gE67
author: 古希腊掌管代码的神
upload_date: 2026-09-18
created: 2026-09-26
tags:
  - bilibili
  - 训练监控
  - WandB
---


> [!info] 视频信息
> **作者**：古希腊掌管代码的神 ｜ **发布日期**：2026-09-18
> **来源**：<https://www.bilibili.com/video/BV1gobj6gE67/>
> **适合人群**：做大模型训练、准备训练岗面试的同学

<iframe src="https://player.bilibili.com/player.html?aid=117219941947412&bvid=BV1gobj6gE67&cid=41620210608&page=1&autoplay=0" scrolling="no" border="0" frameborder="no" framespacing="0" allow="fullscreen; picture-in-picture" allowfullscreen="true" style="height:100%;width:100%; aspect-ratio: 16 / 9;"> </iframe>

> [!tip] 一句话总结
> 训练大模型只看 loss 远远不够。==梯度范数、学习率、吞吐量、显存==这些都是必须监控的。本文讲透怎么搭建一套完整的训练监控体系。

---

## 一、为什么要系统监控

大模型训练是一场**长跑**，动辄几天几周。如果只看 loss，你可能很久之后才发现问题。

> [!quote] 类比
> 系统监控就像汽车仪表盘：速度、油量、温度一目了然，问题出现时第一时间报警。

---

## 二、WandB vs TensorBoard 怎么选

| 维度 | WandB | TensorBoard |
| --- | --- | --- |
| 部署形态 | 云端服务 | 本地工具 |
| 核心优势 | 团队协作、自动对比多组实验、支持告警 | 轻量快速、不需要网络 |
| 适用场景 | 团队协作 | 个人使用 |

> [!note] 选型结论
> - 个人使用 ==TensorBoard 就够了==
> - 团队协作用 ==WandB 更方便==
> - 两者可以**同时使用**：TensorBoard 看实时，WandB 做记录

---

## 三、必须监控的六大指标

| # | 指标 | 正常表现 | 异常信号 |
| --- | --- | --- | --- |
| 1 | **训练 loss** | 整体趋势下降 | 突然飙升 → ==要报警== |
| 2 | **验证 loss** | 与训练 loss 趋势一致 | 训练降、验证升 → 过拟合 |
| 3 | **梯度范数** | 稳定范围内波动 | 突然飙升 → 梯度爆炸 |
| 4 | **学习率实际值** | warmup 与 decay 按预期执行 | 与配置不符 |
| 5 | **吞吐量**（tokens per second） | 稳定 | 突然下降 → 有 IO 瓶颈 |
| 6 | **显存使用** | 留有余量 | 接近上限 → OOM 风险 |

> [!warning] 重点提醒
> 梯度范数是最容易被忽略、却最能提前暴露训练崩溃的信号。稳定波动是正常的，==突然飙升就是梯度爆炸==。

---

## 四、WandB 具体怎么用

1. `pip install wandb`
2. `wandb.init` 加上 project 名和 run 名
3. 在训练循环里用 `wandb.log` 记录指标
4. 在 WandB 界面看实时曲线
5. 设置 alert，当指标超过阈值时发通知

> [!example] 代码骨架
> ```python
> import wandb
>
> wandb.init(project="my-project", name="lr1e-4_bs512")
>
> # 训练循环内
> wandb.log({
>     "train/loss": loss,
>     "val/loss": val_loss,
>     "grad_norm": grad_norm,
>     "lr": lr,
>     "throughput": tokens_per_sec,
>     "gpu/mem": mem_used,
> })
> ```
> `wandb.log` 接收字典键值对，**每个键对应一条曲线**。

---

## 五、实操建议

- [ ] **同时开 WandB 和 TensorBoard**
- [ ] **六大指标全部记录**，不要只记 loss
- [ ] **设置告警阈值**：梯度范数超过「均值 + 3 倍标准差」就报警
- [ ] **定期保存生成样本**，人工眼检判断质量
- [ ] **保存实验配置**（config 全量入库），方便复现
- [ ] **实验结束后用 WandB 自动对比多组实验**，找最优

> [!success] 落地要点
> - 开心实验第一件事是 `wandb.init`，==加上把 config 全量入库==
> - 实验命名里带上关键超参，否则跑几十组之后你分不清谁是谁
> - 每一步记录六大件：训练 loss、验证指标、grad norm、学习率、吞吐、显存占用
> - 不用一直盯前几百步，==密切看稳定之后，每隔几小时扫一眼趋势就够==
