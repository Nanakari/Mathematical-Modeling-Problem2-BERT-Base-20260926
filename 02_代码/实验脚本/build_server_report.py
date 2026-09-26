"""Report matched server ablations with paired video-level uncertainty."""
import csv
import json
from pathlib import Path

import numpy as np

LANE = Path(__file__).resolve().parent
VERIFY = LANE / "server_experiment_verification.json"
PREDICTIONS = LANE / "controlled_valid_server_all_per_sample.csv"
QUICK_ROOT = LANE / "server_quick_results/outputs"
PAIRS = (
    ("加入音频（无视觉）", "text_audio", "text_only"),
    ("加入视觉（无音频）", "text_vision", "text_only"),
    ("已有视觉时加入音频", "full", "text_vision"),
    ("已有音频时加入视觉", "full", "text_audio"),
    ("保留 AV 直连", "full", "no_av_direct"),
    ("保留缺失增广", "full", "no_missing_aug"),
)


def main():
    verify = json.loads(VERIFY.read_text(encoding="utf-8"))
    assert verify["same_data_scaler_source_and_backbone"]
    with PREDICTIONS.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 728 and len({row["sample_id"] for row in rows}) == 728
    video = [row["sample_id"].split("$_$")[0] for row in rows]
    unique_video = list(dict.fromkeys(video))
    assert len(unique_video) == 239
    group = np.array([unique_video.index(item) for item in video], dtype=int)
    counts = np.bincount(group, minlength=len(unique_video))
    rng = np.random.default_rng(101)
    draws = rng.integers(0, len(unique_video), size=(2000, len(unique_video)))
    pair_rows = []
    for label, on, off in PAIRS:
        accuracy_diff = np.array([float(row[f"{on}_correct"]) - float(row[f"{off}_correct"])
                                  for row in rows])
        mae_benefit = np.array([float(row[f"{off}_absolute_error"]) -
                                float(row[f"{on}_absolute_error"]) for row in rows])
        grouped_accuracy = np.bincount(group, weights=accuracy_diff, minlength=len(unique_video))
        grouped_mae = np.bincount(group, weights=mae_benefit, minlength=len(unique_video))
        denominators = counts[draws].sum(axis=1)
        boot_accuracy = grouped_accuracy[draws].sum(axis=1) / denominators
        boot_mae = grouped_mae[draws].sum(axis=1) / denominators
        pair_rows.append({"comparison": label, "on": on, "off": off,
                          "accuracy_difference": float(accuracy_diff.mean()),
                          "accuracy_ci95_low": float(np.quantile(boot_accuracy, 0.025)),
                          "accuracy_ci95_high": float(np.quantile(boot_accuracy, 0.975)),
                          "mae_improvement": float(mae_benefit.mean()),
                          "mae_ci95_low": float(np.quantile(boot_mae, 0.025)),
                          "mae_ci95_high": float(np.quantile(boot_mae, 0.975))})
    with (LANE / "server_controlled_pairwise_bootstrap.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    quick_rows = []
    for name in ("full", "text_only", "text_audio", "text_vision", "no_av_direct", "no_missing_aug"):
        directory = (QUICK_ROOT / "bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729"
                     if name == "full" else QUICK_ROOT / "bert_base_paper_20260926" /
                     f"{name}_seed1729")
        quick = json.loads((directory / "validation_quick.json").read_text(encoding="utf-8"))
        for key in ("accuracy", "macro_f1", "mae", "pearson"):
            assert abs(quick["complete"][key] - verify["runs"][name]["metrics"][key]) < 1e-6
        values = [quick["missing_all_random"][f"all|random|{rate}"]["accuracy"]
                  for rate in (0.1, 0.5, 0.7)]
        quick_rows.append({"name": name, "accuracy_rate_0_1": values[0],
                           "accuracy_rate_0_5": values[1],
                           "accuracy_rate_0_7": values[2],
                           "mean_accuracy": float(np.mean(values))})
    with (LANE / "server_controlled_missing_quick.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quick_rows[0]))
        writer.writeheader()
        writer.writerows(quick_rows)
    f4 = lambda x: f"{x:.4f}"
    lines = ["# 同一 BERT-Base 骨干的服务器受控消融与三种子复核", "",
             "八次训练均在同一服务器 RTX 4090 D、相同代码和 BERT-Base 预训练权重上执行。训练只使用附件 2 train=3395，验证集为 valid=728；标准化只拟合训练集。配置核对表明四个分支方案及关闭缺失增广方案相对完整模型各只更改一个指定因素；除随机种子实验外均用 seed=1729。全部检查点按完整验证集 Accuracy、Macro-F1、MAE 的原顺序选取，未使用测试标签。", "",
             "服务器完整模型 seed 1729 的 Accuracy 为 0.6442；此前附件 3 预测和 113 格缺失压力测试使用的是已冻结的本地检查点，其 Accuracy 为 0.6511。两者是相同配置在不同硬件上的独立训练结果，不能把服务器消融表与本地 113 格表当成同一个检查点的结果。", "",
             "## 三种子完整模型", "",
             "| seed | 完成轮次 | 最佳轮次 | Accuracy | Macro-F1 | MAE | Pearson |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for name, seed in (("full", 1729), ("full_seed2718", 2718), ("full_seed3407", 3407)):
        item = verify["runs"][name]
        m = item["metrics"]
        lines.append(f"| {seed} | {item['completed_epochs']} | {item['best_epoch']} | {f4(m['accuracy'])} | {f4(m['macro_f1'])} | {f4(m['mae'])} | {f4(m['pearson'])} |")
    lines += ["", "| 指标 | 均值 | 样本标准差 |", "|---|---:|---:|"]
    for key in ("accuracy", "macro_f1", "mae", "pearson", "neutral_recall"):
        stat = verify["three_seed_statistics"][key]
        lines.append(f"| {key} | {f4(stat['mean'])} | {f4(stat['sample_sd'])} |")
    lines += ["", "## 同种子受控方案（完整验证集）", "",
              "BERT-only、BERT+音频和 BERT+视觉是在相同网络骨架内，将未使用模态的输入值和观测掩码同时置零后重新训练；因此这里衡量输入信息贡献，并非减少参数量后的轻量模型。", "",
              "| 方案 | 完成轮次 | 最佳轮次 | Accuracy | Macro-F1 | MAE | Pearson |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("text_only", "text_audio", "text_vision", "full", "no_av_direct", "no_missing_aug"):
        item = verify["runs"][name]
        m = item["metrics"]
        lines.append(f"| {name} | {item['completed_epochs']} | {item['best_epoch']} | {f4(m['accuracy'])} | {f4(m['macro_f1'])} | {f4(m['mae'])} | {f4(m['pearson'])} |")
    lines += ["", "## 逐样本配对差值", "",
              "正的 Accuracy 差值和 MAE 改善量表示前一方案更好；负值表示更差。区间用 239 个原始视频为抽样单位做 2000 次配对 bootstrap，仅描述这一个验证集的抽样不确定性。", "",
              "| 比较 | ΔAccuracy | 95% 区间 | MAE 改善量 | 95% 区间 |",
              "|---|---:|---:|---:|---:|"]
    for row in pair_rows:
        lines.append(f"| {row['comparison']} | {row['accuracy_difference']:+.4f} | [{f4(row['accuracy_ci95_low'])}, {f4(row['accuracy_ci95_high'])}] | {row['mae_improvement']:+.4f} | [{f4(row['mae_ci95_low'])}, {f4(row['mae_ci95_high'])}] |")
    lines += ["", "## 共同的随机局部缺失情景", "",
              "六个检查点使用相同验证集、随机种子 101、三模态同时缺失、随机位置和名义比例 0.1/0.5/0.7。下表为 Accuracy，详细指标保存在各验证 JSON 中。", "",
              "| 方案 | 比例 0.1 | 比例 0.5 | 比例 0.7 | 三情景均值 |",
              "|---|---:|---:|---:|---:|"]
    for row in quick_rows:
        lines.append(f"| {row['name']} | {f4(row['accuracy_rate_0_1'])} | {f4(row['accuracy_rate_0_5'])} | {f4(row['accuracy_rate_0_7'])} | {f4(row['mean_accuracy'])} |")
    full_quick = next(row for row in quick_rows if row["name"] == "full")
    no_aug_quick = next(row for row in quick_rows if row["name"] == "no_missing_aug")
    no_direct_quick = next(row for row in quick_rows if row["name"] == "no_av_direct")
    lines += ["", f"完整模型相对关闭缺失增广的三情景平均 Accuracy 高 {full_quick['mean_accuracy']-no_aug_quick['mean_accuracy']:.4f}；在比例 0.7 时高 {full_quick['accuracy_rate_0_7']-no_aug_quick['accuracy_rate_0_7']:.4f}。完整模型相对移除 AV 直连的三情景平均 Accuracy 高 {full_quick['mean_accuracy']-no_direct_quick['mean_accuracy']:.4f}，但其完整输入 Accuracy 更低，说明 AV 直连呈现干净输入与缺失鲁棒性的取舍。", "",
              "这些差值并不都为正。音频、视觉的边际差值依赖另一模态是否存在；不能笼统声称每个分支都有增益。缺失增广对照采用相同选模与早停规则，轮数自然不同；单种子差值仍不代表跨种子稳定效应。", "",
              "逐样本结果见 `controlled_valid_server_all_per_sample.csv`，包含六方案对同一 728 条验证样本的类别、概率、强度、分类正确标记及绝对误差。配置与检查点哈希核对见 `server_experiment_verification.json`。服务器检查点保存在 `/home/a631/gaotianchang/bert_base_controlled_20260926`，未重新评估官方测试集或更换附件 3 的冻结提交模型。", ""]
    (LANE / "服务器同骨干消融与三种子复核.md").write_text("\n".join(lines), encoding="utf-8")
    print(LANE / "服务器同骨干消融与三种子复核.md")


if __name__ == "__main__":
    main()
