"""Physically reorder bibitems so that keys 10,11,12 come after key 9"""
import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all bibitem entries with their full text
# Pattern: \bibitem{KEY}...TEXT... until next \bibitem or \end{thebibliography}
pattern = r'(\\bibitem\{(\d+)\}.*?)(?=\\bibitem\{\d+\}|\\end\{thebibliography\})'
matches = list(re.finditer(pattern, content, re.DOTALL))

entries = []
for m in matches:
    key = int(m.group(2))
    text = m.group(1)
    entries.append((key, text))

print(f"Found {len(entries)} bibitems")
for k, t in entries[:25]:
    short = t[:60].replace('\n', ' ')
    print(f"  key={k}: {short}...")

# Find the boundary after bibitem{9}
after_9 = -1
for i, (k, t) in enumerate(entries):
    if k == 9:
        after_9 = i
        break

# Find entries 10, 11, 12
entries_to_move = {}
pos_10 = -1
for i, (k, t) in enumerate(entries):
    if k in (10, 11, 12):
        entries_to_move[k] = (i, t)
        if pos_10 < 0:
            pos_10 = i

if not entries_to_move:
    print("ERROR: keys 10,11,12 not found")
    exit(1)

print(f"\nEntries 10,11,12 found at positions {pos_10}+")
print(f"Moving them to after position {after_9} (key=9)")

# Rebuild: keep 0..after_9, add 10,11,12, then the rest
new_order = []
for i, (k, t) in enumerate(entries):
    if k in (10, 11, 12):
        continue  # skip (will add later)
    new_order.append(t)
    if k == 9:
        # Insert 10,11,12 here
        for kk in (10, 11, 12):
            new_order.append(entries_to_move[kk][1])

# Replace the bibliography section
start_marker = r'\begin{thebibliography}'
end_marker = r'\end{thebibliography}'
start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx < 0 or end_idx < 0:
    print("ERROR: bibliography markers not found")
    exit(1)

new_bib = '\\begin{thebibliography}{43} \\vskip 7pt\n'
new_bib += '\n'.join(new_order)
new_bib += '\n\n\\end{thebibliography}'

old_bib = content[start_idx:end_idx + len(end_marker)]
content = content.replace(old_bib, new_bib)

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone! Bibliography reordered.")
print("New order: 1..9, 10(MambaPolicy), 11(DecisionMamba), 12(TimeMachine), 13(Dreamer)...")
