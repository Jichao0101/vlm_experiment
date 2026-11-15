"""生成并顺序执行 vision SDPA/FA2 对照，复用样本测量入口。"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    source_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=source_dir / 's02_sample_eval.json',
                        help='基础样本配置；继承预热、输入和生成预算')
    parser.add_argument('--model', type=Path, help='覆盖基础配置的模型路径')
    parser.add_argument('--sample-manifest', type=Path, help='覆盖基础配置的样本清单')
    parser.add_argument('--out', type=Path, required=True, help='本轮新目录，保存配置与结果')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--modes', nargs='+', choices=['operator', 'timing', 'memory', 'benchmark'],
                        default=['operator', 'timing', 'benchmark'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--capture-batches', type=int, default=1)
    parser.add_argument('--reverse', action='store_true', help='每个模式先运行 FA2，再运行 SDPA')
    parser.add_argument('--prepare-only', action='store_true', help='仅生成配置及打印命令')
    args = parser.parse_args()
    if args.batch_sizes != sorted(set(args.batch_sizes)) or min(args.batch_sizes) < 1:
        parser.error('--batch-sizes 必须为递增的正整数且无重复')
    if min(args.repeats, args.capture_batches) < 1:
        parser.error('--repeats 和 --capture-batches 必须为正整数')
    if len(args.modes) != len(set(args.modes)):
        parser.error('--modes 不应重复')
    if not args.gpu or ',' in args.gpu:
        parser.error('--gpu 只接受单个 GPU ID 或 UUID')

    config_path = args.config.expanduser().resolve()
    base = json.loads(config_path.read_text(encoding='utf-8'))
    # 原配置的相对路径基于原 JSON；CLI 覆盖路径基于调用目录。
    for key in ('model', 'sample_manifest'):
        override = getattr(args, key)
        path = override if override is not None else Path(base[key]).expanduser()
        if override is None and not path.is_absolute():
            path = config_path.parent / path
        base[key] = str(path.expanduser().resolve())
    base.update(batch_sizes=args.batch_sizes, gpus=[args.gpu])
    for key in ('repeats', 'capture_batches', 'dp_batch_size'):
        base.pop(key, None)
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f'输出目录已存在，请使用新目录：{out}')

    # 正式执行前检查候选接口，避免测完基线后才发现 FA2 缺依赖。
    if not args.prepare_only:
        subprocess.run([sys.executable, '-c',
                        'import torch, transformers, flash_attn; '
                        'from flash_attn import flash_attn_varlen_func; '
                        'print(torch.__version__, torch.version.cuda, '
                        'transformers.__version__, flash_attn.__version__)'], check=True)
    out.mkdir(parents=True, exist_ok=False)
    config_dir = out / 'configs'
    config_dir.mkdir()
    backends = [('sdpa', 'sdpa'), ('fa2', 'flash_attention_2')]
    if args.reverse:
        backends.reverse()
    commands = []
    for mode in args.modes:
        for label, backend in backends:
            config = dict(base)
            config.update(mode=mode, out=str(out / f'{label}_{mode}'),
                          attn_implementation={'vision_config': backend, 'text_config': 'sdpa'})
            if mode == 'benchmark':
                config['repeats'] = args.repeats
            else:
                config['capture_batches'] = args.capture_batches
            path = config_dir / f'{label}_{mode}.json'
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            commands.append([sys.executable, str(source_dir / 's02_sample_eval.py'), '--config', str(path)])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.prepare_only:
            # 子进程非零退出即停止，保留本轮配置和已有结果供定位或单项重跑。
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
