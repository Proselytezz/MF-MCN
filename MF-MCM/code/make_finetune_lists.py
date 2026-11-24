import os
import random

base = r'E:\MF-MCM-main\data\BraTS2023_GLI_01scale'

# 从预处理目录枚举 1251 例，避免依赖会被覆盖的 txt
names = sorted(
    d for d in os.listdir(base)
    if d.startswith('BraTS-GLI-') and os.path.isdir(os.path.join(base, d))
)
assert len(names) == 1251, f'expect 1251 cases, got {len(names)}'

random.seed(1337)
random.shuffle(names)

# 论文划分：预训练 900 例，其余 351 例按 210/70/71 微调
pretrain, rest = names[:900], names[900:]
random.shuffle(rest)
train, val, test = rest[:210], rest[210:280], rest[280:351]

# 防止预训练/微调/验证/测试之间出现泄漏
assert len(pretrain) == 900
assert len(train) == 210 and len(val) == 70 and len(test) == 71
assert not (set(pretrain) & set(train))
assert not (set(pretrain) & set(val))
assert not (set(pretrain) & set(test))
assert not (set(train) & set(val)) and not (set(train) & set(test))
assert not (set(val) & set(test))

for fn, lst in [('pretrain_path_list.txt', pretrain),
                ('train_path_list.txt', train),
                ('val_path_list.txt', val),
                ('test_path_list.txt', test)]:
    with open(os.path.join(base, fn), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lst) + '\n')

print(f'pretrain: {len(pretrain)}, train: {len(train)}, '
      f'val: {len(val)}, test: {len(test)}')
