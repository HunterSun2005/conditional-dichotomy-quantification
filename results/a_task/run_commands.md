# 成员 A 运行命令

生成成员 A 的全部产物：

```bash
python scripts/a_task/generate_member_a_artifacts.py
```

使用不同随机种子重新生成：

```bash
python scripts/a_task/generate_member_a_artifacts.py --seed 123
```

查看已生成的标准化子集：

```bash
ls data/member_a_splits/*/*/{clean,hard,small_500}.jsonl
```

当 DoGE/Opposite-Score checkpoint 可用时，使用 `data/member_a_splits/context_variants/` 中的 context 扰动文件，接入官方 evaluator 或 B 的统一评测脚本重新运行 E8。记录时需要保留模型路径、checkpoint、seed、batch size 和指标结果，并写入 `results/a_task/context_sensitivity_results.csv` 或全组共享的实验登记表。
