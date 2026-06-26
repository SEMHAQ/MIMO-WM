"""删除重复引用，只保留第一次出现"""
import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 找出所有 \cite{...} 引用
cite_pattern = r'\\cite\{([^}]+)\}'
matches = list(re.finditer(cite_pattern, content))

# 追踪已引用的参考文献
cited_refs = set()
segments = []
last_end = 0
total_removed = 0

for match in matches:
    refs_str = match.group(1)
    refs = [r.strip() for r in refs_str.split(',')]

    # 检查哪些引用是新的
    new_refs = [r for r in refs if r not in cited_refs]

    # 添加引用前的文本
    segments.append(content[last_end:match.start()])

    if new_refs:
        # 有新引用，保留整个引用
        for r in new_refs:
            cited_refs.add(r)
        segments.append(match.group(0))
    else:
        # 全部是重复引用，删除整个 \textsuperscript{\cite{...}}
        # 需要回溯找到 \textsuperscript{
        prefix = content[:match.start()]
        if prefix.endswith('\\textsuperscript{'):
            # 删除 \textsuperscript{
            segments[-1] = segments[-1][:-len('\\textsuperscript{')]
        total_removed += 1

    last_end = match.end()

# 添加剩余文本
segments.append(content[last_end:])

new_content = ''.join(segments)

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f'处理完成:')
print(f'  删除重复引用: {total_removed} 处')
print(f'  保留引用: {len(cited_refs)} 个')
print(f'  引用列表: {sorted(cited_refs, key=lambda x: int(x) if x.isdigit() else x)}')
