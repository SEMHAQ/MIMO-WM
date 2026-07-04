"""Reorder bibitem numbers to fix citation order"""
import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Mapping: old_number -> new_number
# Current order: 1-9, 20-22, 10-19, 23-29
# Fix: 1-9, 10-12(=old 20-22), 13-19(=old 10-16), 17(old)-28(new 23-28), 29 unchanged
# Actually let me just trace through the first appearance order carefully
# and make it sequential

# The mapping:
# old 1-9 → 1-9 (no change)
# old 20 → 10 (Mamba Policy)
# old 21 → 11 (Decision Mamba)
# old 22 → 12 (TimeMachine)
# old 10 → 13 (Dreamer)
# old 11 → 14 (DreamerV3)
# old 12 → 15 (Nagabandi)
# old 13 → 16 (Zhang MPC)
# old 14 → 17 (Wang MPC)
# old 15 → 18 (TinyML)
# old 16 → 19 (Hu survey)
# old 17 → 20 (HiPPO)
# old 18 → 21 (S5)
# old 19 → 22 (H3)
# old 23 → 23 (RoboMamba) - same
# old 24 → 24 (PETS) - same
# old 25 → 25 (Wang EMPC) - same
# old 26 → 26 (GELU) - same
# old 27 → 27 (AdamW) - same
# old 28 → 28 (MuJoCo) - same
# old 29 → 29 (Gymnasium) - same

mapping = {
    20: 10, 21: 11, 22: 12,
    10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19, 17: 20, 18: 21, 19: 22,
}

# First, update all \cite{...} calls
def replace_cite(match):
    full = match.group(0)
    inner = match.group(1)
    # Parse comma-separated numbers
    nums = inner.split(',')
    new_nums = []
    for n in nums:
        n = n.strip()
        if n.isdigit():
            old = int(n)
            new_nums.append(str(mapping.get(old, old)))
        else:
            new_nums.append(n)
    return f'\\cite{{{",".join(new_nums)}}}'

content = re.sub(r'\\cite\{([^}]+)\}', replace_cite, content)

# Then, update all \bibitem{...} entries
def replace_bibitem(match):
    full = match.group(0)
    num = match.group(1)
    if num.isdigit():
        old = int(num)
        new_num = mapping.get(old, old)
        return f'\\bibitem{{{new_num}}}'
    return full

content = re.sub(r'\\bibitem\{(\d+)\}', replace_bibitem, content)

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
print("bibitem renumbering complete")
for old, new in sorted(mapping.items()):
    print(f"  {old} → {new}")
