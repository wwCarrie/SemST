"""
Embryo(E1S1) 误差分析

用法:
    python error_analysis/run_Embryo_E1S1.py
前置:
    python preprocess/Embryo_generate_data.py   # 已生成 generate_data/Embryo/E1S1/data.h5ad
"""
from common import analyze

DATASET_ID = 'E1S1'

if __name__ == '__main__':
    r = analyze(DATASET_ID)
    print("=" * 60)
    print(f"[{r['config_name']}/{r['dataset_id']}] "
          f"ARI={r['ari']:.2f} NMI={r['nmi']:.2f} ACC={r['acc']:.2f} "
          f"F1w={r['f1w']:.2f} F1m={r['f1m']:.2f}")
    print(f"类别数={r['class_num']}  spot数={r['N']}")
    large = f"{r['large_acc']:.1f}%" if r['large_acc'] is not None else "N/A"
    small = f"{r['small_acc']:.1f}%" if r['small_acc'] is not None else "N/A"
    print(f"规模-准确率相关系数={r['size_acc_corr']:.3f}, "
          f"大类别均acc={large}, 小类别均acc={small}")
    print(f"边界邻接混淆率={r['boundary_confusion_rate']:.1f}%")
    print(f"输出目录: {r['out_dir']}")
