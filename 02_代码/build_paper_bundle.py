"""Build the self-contained paper-writing bundle for Problem 2 BERT-Base."""
from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

PROJECT = Path(r"D:\codex_projects\projects\Mathematical Modeling")
P2 = PROJECT / "problem2"
E = PROJECT / "E题"
LANE = P2 / "outputs/bert_base_paper_20260926"
DEST = PROJECT / "问题2_BERT-Base_论文资料包_20260926"

copied: list[tuple[Path, Path]] = []


def add(source: Path, relative: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.append((source, target))


def add_tree(source: Path, relative: str, *, ignore: set[str] | None = None) -> None:
    ignore = ignore or set()
    for file in sorted(source.rglob("*")):
        if file.is_file() and not any(part in ignore for part in file.relative_to(source).parts):
            add(file, str(Path(relative) / file.relative_to(source)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    add(E / "复杂场景下多模态情感识别的数学建模与算法设计.docx", "01_赛题与数据/赛题原文.docx")
    add(E / "E题数据/附件2-数据集特征文件/aligned_50.pkl", "01_赛题与数据/附件2_aligned_50.pkl")
    add_tree(E / "E题数据/附件3-模态缺失特征样本/对齐版本", "01_赛题与数据/附件3_对齐版")
    add(P2 / "outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl", "01_赛题与数据/train_valid_only_v5.pkl")

    for name in ("v5_data.py", "v5_export.py", "v5_model.py", "v5_run.py", "v5_train.py",
                 "aligned_dataset.py", "p2.py", "pipeline.py", "requirements-v5.txt",
                 "test_v5_contracts.py", "test_v5_export.py", "verify_tokenizer.py"):
        add(P2 / name, f"02_代码/{name}")
    for file in sorted(LANE.iterdir()):
        if file.is_file() and file.suffix in {".py", ".sh", ".ps1"} and file.name != "build_paper_bundle.py":
            add(file, f"02_代码/实验脚本/{file.name}")
    add(LANE / "build_paper_bundle.py", "02_代码/build_paper_bundle.py")
    add(P2 / "configs/v5/student_bert_base_retest_seed1729.json", "02_代码/配置/student_bert_base_retest_seed1729.json")
    for file in sorted(LANE.glob("*seed*.json")):
        add(file, f"02_代码/配置/{file.name}")

    add_tree(P2 / "models/teacher_student_v5_20260924/bert_base_uncased",
             "03_模型/预训练BERT-Base", ignore={".cache"})
    frozen = P2 / "outputs/bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729"
    for name in ("best_model.safetensors", "best_validation.json", "candidate_summary.json",
                 "prepared_manifest.json", "pretrained_provenance.json", "protocol.json",
                 "scaler.json", "source_hashes.json", "training_history.jsonl"):
        add(frozen / name, f"03_模型/最终冻结检查点/{name}")
    add(P2 / "outputs/bert_base_vs_minilm_20260926/package_student_bert_base_retest_seed1729.zip",
        "03_模型/最终推理包.zip")

    for file in sorted(LANE.iterdir()):
        if file.is_file() and file.suffix in {".csv", ".json", ".md"} and "seed" not in file.stem:
            add(file, f"04_实验结果/论文分析/{file.name}")
    for folder in ("server_replication_snapshot", "server_quick_results"):
        add_tree(LANE / folder, f"04_实验结果/{folder}", ignore={"__pycache__"})
    base_results = P2 / "outputs/bert_base_vs_minilm_20260926"
    for name in ("grid_student_bert_base_retest_seed1729.json", "grid_student_bert_base_retest_seed1729.npz",
                 "student_bert_base_retest_seed1729_official_test.json",
                 "student_bert_base_retest_seed1729_attachment3_predictions.csv",
                 "freeze_student_bert_base_retest_seed1729.json",
                 "student_bert_base_retest_seed1729_freeze_pre_official_test_snapshot.json",
                 "student_bert_base_retest_seed1729_package_validation_v4_32.json",
                 "问题2_缺失影响规律分析.md"):
        add(base_results / name, f"04_实验结果/本地最终模型/{name}")
    add(P2 / "outputs/two_model_rerun_20260926/bert_base_v5_attachment3_predictions.csv",
        "04_实验结果/附件3_30条最终预测.csv")
    add(P2 / "outputs/two_model_rerun_20260926/bert_base_v5_validation_32.json",
        "04_实验结果/本地最终模型/bert_base_v5_validation_32.json")

    with (DEST / "文件清单.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["归档相对路径", "字节数", "SHA256", "原始路径"])
        for source, target in copied:
            writer.writerow([str(target.relative_to(DEST)), target.stat().st_size, sha256(target), str(source)])
    total = sum(target.stat().st_size for _, target in copied)
    (DEST / "README_资料包说明.md").write_text(
        "# 问题（2）BERT-Base 论文资料包\n\n"
        "本文件夹只收录撰写问题（2）答题论文及复核现有结论所需资料，不是可直接提交的竞赛附件。\n\n"
        "- `01_赛题与数据`：赛题原文、附件2对齐版、附件3对齐版30条样本，以及训练专用的train/valid文件。\n"
        "- `02_代码`：BERT-Base模型、训练和推理代码、配置、实验分析脚本。\n"
        "- `03_模型`：预训练BERT-Base、本地选定的最终冻结检查点及推理包。\n"
        "- `04_实验结果`：113格缺失实验、验证集错误归因、三种子分支消融、附件3预测及测试诊断。\n"
        "- `文件清单.csv`：每个归档文件的大小、SHA256和来源。\n\n"
        "论文中的最终模型是本地冻结的BERT-Base seed 1729；服务器同配置模型是独立训练的消融对照。"
        "服务器其他种子的完整权重和训练中间状态未收录，留在服务器 `/home/a631/gaotianchang/bert_base_controlled_20260926`。"
        "三种子测试集结果属于重复访问测试集后的事后诊断，不得当作新盲测或用于选模。"
        "附件3的特征零值不等于真实缺失标注；缺失长度以对齐位置数表示，不能写成秒。"
        "当前模型包超过赛题50MB总附件限制，正式提交前需要另行处理。\n\n"
        f"归档文件数：{len(copied)}；总大小：{total:,} 字节。\n",
        encoding="utf-8")
    print(DEST)
    print(f"files={len(copied)} bytes={total}")


if __name__ == "__main__":
    main()
