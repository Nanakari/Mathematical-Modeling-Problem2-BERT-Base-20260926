"""Summarize the two selected seeds and the matched augmentation ablation."""
import csv
import json
import statistics
from pathlib import Path

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[1]
OLD = ROOT / "outputs/bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729"
METRICS = ("accuracy", "macro_f1", "mae", "pearson", "neutral_recall")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def result(name, output, seed, quick_file):
    best = read(output / "best_validation.json")
    quick = read(output / quick_file)
    for key in METRICS:
        assert abs(best["metrics"][key] - quick["complete"][key]) < 1e-6
    cells = quick["missing_all_random"]
    return {"name": name, "seed": seed, "best_epoch": best["epoch"],
            **{key: best["metrics"][key] for key in METRICS},
            "missing_random_mean_accuracy": statistics.mean(
                cells[f"all|random|{rate}"]["accuracy"] for rate in (0.1, 0.5, 0.7))}


def main():
    full = [result("full_seed1729", OLD, 1729, "source_validation_v4_32.json"),
            result("full_seed2718", LANE / "full_seed2718", 2718, "validation_quick.json")]
    no_aug = result("no_missing_aug_seed1729", LANE / "no_missing_aug_seed1729", 1729,
                    "validation_quick.json")
    rows = full + [no_aug]
    with (LANE / "experiment_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    f4 = lambda value: f"{value:.4f}"
    lines = ["# BERT-Base 问题 2：两种子与缺失增强配对实验", "",
             "主模型固定为 BERT-Base。两个完整模型均训练了 6 轮；seed 1729 按早停规则结束，seed 2718 按用户要求在第 6 轮后停止，采用第 5 轮的最佳检查点。该检查点在 728 条验证集上复算指标与保存记录完全一致。两种子统计样本量很小，seed 2718 也未走到原早停条件，不能据此作稳定性显著结论。", "",
             "## 完整验证集", "",
             "| 种子 | 最佳轮次 | Accuracy | Macro-F1 | MAE | Pearson | 中性类召回率 |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in full:
        lines.append(f"| {row['seed']} | {row['best_epoch']} | {f4(row['accuracy'])} | {f4(row['macro_f1'])} | {f4(row['mae'])} | {f4(row['pearson'])} | {f4(row['neutral_recall'])} |")
    lines += ["", "| 指标 | 两种子均值 | 样本标准差 |", "|---|---:|---:|"]
    for key in METRICS:
        values = [row[key] for row in full]
        lines.append(f"| {key} | {f4(statistics.mean(values))} | {f4(statistics.stdev(values))} |")
    lines += ["", "## 缺失增强配对重训（seed 1729）", "",
              "配置仅将训练视图从 mixed 改为 full；其余模型、数据、种子、优化参数及验证集选模规则相同。原模型完成 6 轮，本配对实验按用户指定条件在第 4 轮后停止，采用第 2 轮最佳检查点；实际训练轮数不同。", "",
              "| 方案 | Accuracy | Macro-F1 | MAE | Pearson | 三个随机缺失情景平均 Accuracy |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in (full[0], no_aug):
        lines.append(f"| {row['name']} | {f4(row['accuracy'])} | {f4(row['macro_f1'])} | {f4(row['mae'])} | {f4(row['pearson'])} | {f4(row['missing_random_mean_accuracy'])} |")
    lines += ["", "随机缺失情景为三模态同时在随机位置按名义比例 0.1、0.5、0.7 遮蔽。关闭缺失增强后，完整验证集 Accuracy 和三个缺失情景平均 Accuracy 均低 "
              f"{f4(full[0]['accuracy'] - no_aug['accuracy'])}。配对重训只有一个随机种子且实际训练轮数不同，差值仅作为初步机制证据。", "",
              "## 比赛要求的最小核对", "",
              "- 训练文件只含 train 3395 和 valid 728；训练/验证视频 ID 不重叠。本轮训练与验证未使用测试标签。",
              "- 已有的 BERT-Base 验证集 113 格缺失分析覆盖缺失模态类型、位置和名义比例；附件 3 已有 30 条极性与强度预测。",
              "- 对齐特征没有物理时间戳；缺失时长应写为对齐位置长度，名义遮蔽比例不等于实际移除的有效位置比例。",
              "- 历史官方测试指标在模型冻结后生成，但项目更早已访问过该测试集；论文不得称其为完全盲测。本轮不依据测试结果换模型。",
              "- 本轮按用户要求暂不处理 50 MB 提交上限；现有 BERT-Base 压缩包约 202 MB，若直接提交则不符合这一条。", ""]
    output = LANE / "BERT-Base_两种子与流程核对.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
