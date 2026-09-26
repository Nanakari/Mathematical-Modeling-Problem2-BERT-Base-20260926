"""Merge the six matched BERT-Base validation predictions by sample ID."""
import csv
import json
from pathlib import Path

LANE = Path(__file__).resolve().parent
LOCAL = (LANE / "controlled_valid_local_per_sample.csv",
         LANE / "controlled_valid_local_summary.json")
REMOTE = (LANE / "controlled_valid_remote_per_sample.csv",
          LANE / "controlled_valid_remote_summary.json")
ARMS = ("full", "text_only", "text_audio", "text_vision", "no_av_direct", "no_missing_aug")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    local = read_csv(LOCAL[0])
    remote = read_csv(REMOTE[0])
    assert len(local) == len(remote) == 728
    lookup = {row["sample_id"]: row for row in remote}
    assert len(lookup) == 728 and set(lookup) == {row["sample_id"] for row in local}
    rows = []
    for row in local:
        other = lookup[row["sample_id"]]
        assert row["true_class"] == other["true_class"]
        assert abs(float(row["true_intensity"]) - float(other["true_intensity"])) < 1e-8
        rows.append({**row, **{key: value for key, value in other.items()
                              if key not in {"sample_id", "true_class", "true_intensity"}}})
    with (LANE / "controlled_valid_per_sample.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {**json.loads(LOCAL[1].read_text(encoding="utf-8")),
               **json.loads(REMOTE[1].read_text(encoding="utf-8"))}
    assert set(summary) == set(ARMS)
    (LANE / "controlled_valid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fmt = lambda x: f"{x:.4f}"
    full = summary["full"]["metrics"]
    lines = ["# 同一 BERT-Base 骨干的验证集受控消融", "",
             "六个方案使用附件 2 的相同 train/valid 划分和 seed 1729；模型检查点只依据完整验证集 Accuracy、Macro-F1、MAE 的既定顺序选择。BERT-only、BERT+音频、BERT+视觉通过同时屏蔽停用模态的数值与观测掩码重训；移除 AV 直连只改变该结构开关。", "",
             "| 方案 | Accuracy | 相对完整模型 | Macro-F1 | MAE | Pearson | 最佳轮次 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ARMS:
        item = summary[name]
        m = item["metrics"]
        lines.append(f"| {name} | {fmt(m['accuracy'])} | {m['accuracy']-full['accuracy']:+.4f} | {fmt(m['macro_f1'])} | {fmt(m['mae'])} | {fmt(m['pearson'])} | {item['best_epoch']} |")
    pairs = (
        ("加入音频（无视觉）", "text_audio", "text_only"),
        ("加入视觉（无音频）", "text_vision", "text_only"),
        ("在视觉存在时加入音频", "full", "text_vision"),
        ("在音频存在时加入视觉", "full", "text_audio"),
        ("保留 AV 直连", "full", "no_av_direct"),
        ("保留缺失增广", "full", "no_missing_aug"),
    )
    lines += ["", "| 比较 | Accuracy 差值 | Macro-F1 差值 | MAE 改善量 |",
              "|---|---:|---:|---:|"]
    for label, on, off in pairs:
        a, b = summary[on]["metrics"], summary[off]["metrics"]
        lines.append(f"| {label} | {a['accuracy']-b['accuracy']:+.4f} | {a['macro_f1']-b['macro_f1']:+.4f} | {b['mae']-a['mae']:+.4f} |")
    lines += ["", "以上差值是单个随机种子在同一验证集上的描述性结果，不能当作跨种子稳定收益。缺失增广方案按用户要求在第 4 轮后停止，完整模型训练 6 轮，因此该项的训练轮数不同，不能作严格等预算因果结论。", "",
              "逐样本对照见 `controlled_valid_per_sample.csv`：每行含真实标签和六个方案的预测类别、类别概率、预测强度、分类正确标记及强度绝对误差。", ""]
    (LANE / "同骨干受控消融对比.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
