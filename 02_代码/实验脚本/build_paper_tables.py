"""Create concise paper-facing tables from frozen experiment records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANE = Path(__file__).resolve().parent
SNAP = LANE / "server_replication_snapshot"
VERIFIED = json.loads((LANE / "server_experiment_verification.json").read_text(encoding="utf-8"))

arms = ("full", "text_only", "text_audio", "text_vision", "no_av_direct", "no_missing_aug")
labels = {"full": "完整模型", "text_only": "BERT-only", "text_audio": "BERT+音频",
          "text_vision": "BERT+视觉", "no_av_direct": "移除AV直连", "no_missing_aug": "关闭缺失增广"}
rows = []
for seed in (1729, 2718, 3407):
    for arm in arms:
        if arm == "no_missing_aug" and seed != 1729:
            continue
        if seed == 1729:
            run = VERIFIED["runs"][arm]
            metrics = run["metrics"]
            epoch = run["best_epoch"]
        else:
            run = json.loads((SNAP / f"{arm}_seed{seed}" / "best_validation.json").read_text(encoding="utf-8"))
            metrics = run["metrics"]
            epoch = run["epoch"]
        rows.append({"seed": seed, "scheme": arm, "scheme_zh": labels[arm],
                     "best_epoch": epoch,
                     **{key: metrics[key] for key in ("accuracy", "macro_f1", "mae", "pearson")}})

with (LANE / "论文用_统一消融四指标.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

config = json.loads((ROOT / "configs/v5/student_bert_base_retest_seed1729.json").read_text(encoding="utf-8"))
model = config["model"]
train = config["training"]
data = config["data"]
lines = [
    "# 问题（2）BERT-Base 论文用训练配置与指标总表", "",
    "## 固定数据与选模", "",
    "- 附件 2 对齐版：train 3395、valid 728；三种子为 1729、2718、3407。标准化只在 train 拟合。",
    "- 骨干：`google-bert/bert-base-uncased`，固定 revision `" + config["model_revision"] + "`。",
    "- 每次训练只用 train 更新参数；完整 valid 按 Accuracy、Macro-F1、MAE 的原顺序选最佳检查点，patience 为 " + str(train["patience"]) + "。",
    "- 消融只改变指定模态输入/观测掩码、AV 直连开关或缺失增广策略；三种子方案保持相同数据划分及选模规则。",
    "", "## 关键参数", "",
    "| 参数 | 值 |", "|---|---|",
    f"| 文本最长长度 | {data['max_text_length']} |",
    f"| 融合维度 / 时序宽度 | {model['fusion_dim']} / {model['temporal_width']} |",
    f"| 音频/视觉编码层数 | {model['audio_layers']} / {model['vision_layers']} |",
    f"| 池化 / 融合 | {model['pooling']} / {model['fusion']} |",
    f"| 注意力头数 / dropout | {model['num_heads']} / {model['dropout']} |",
    f"| 最大轮数 / batch / 梯度累积 | {train['epochs']} / {train['batch_size']} / {train['gradient_accumulation_steps']} |",
    f"| BERT 学习率 / 任务头学习率 | {train['encoder_lr']} / {train['head_lr']} |",
    f"| 权重衰减 / 梯度裁剪 | {train['weight_decay']} / {train['max_grad_norm']} |",
    f"| 混合精度 / 回归损失权重 | {train['amp_dtype']} / {train['regression_loss_weight']} |",
    f"| 缺失增广 | {train['view_policy']}；完整输入概率 {train['clean_probability']}；训练缺失比例 {train['mask_rates']} |",
    f"| 验证缺失比例 / 掩码种子 | {train['evaluation_mask_rates']} / {train['validation_mask_seed']} |",
    "", "## 同骨干消融：验证集四指标", "",
    "| 种子 | 方案 | 最佳轮次 | Accuracy | Macro-F1 | MAE | Pearson |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(f"| {row['seed']} | {row['scheme_zh']} | {row['best_epoch']} | "
                 f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['mae']:.4f} | {row['pearson']:.4f} |")
lines.extend(["", "关闭缺失增广只有 1729 种子，不能作为跨种子稳定效应。",
              "表中服务器完整模型与本地最终冻结检查点为同配置独立训练，论文的 113 格缺失实验与附件 3 预测属于后者。",
              "服务器三种子测试结果属于重复访问测试集后的事后诊断，不参与选模。", ""])
(LANE / "论文用_训练配置与消融总表.md").write_text("\n".join(lines), encoding="utf-8")
print(f"rows={len(rows)}")
