import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1) SSM continuous: two-line && eqnarray -> aligned
old1 = '\\begin{eqnarray}\n&&\\dot{{\\bm h}}(t) = {\\bm A}{\\bm h}(t) + {\\bm B}{\\bm x}(t),\\\\\n&&{\\bm y}(t) = {\\bm C}{\\bm h}(t) + {\\bm D}{\\bm x}(t),\n\\end{eqnarray}'
new1 = '\\begin{equation}\n\\left\{\\begin{aligned}\n\\dot{{\\bm h}}(t) &= {\\bm A}{\\bm h}(t) + {\\bm B}{\\bm x}(t),\\\\\n{\\bm y}(t) &= {\\bm C}{\\bm h}(t) + {\\bm D}{\\bm x}(t).\n\\end{aligned}\\right.\n\\end{equation}'
if old1 in content:
    content = content.replace(old1, new1)
    print("1) SSM cont: DONE")
else:
    print("1) SSM cont: NOT FOUND")

# 2) SSM discrete: two-line && eqnarray -> aligned
old2 = '\\begin{eqnarray}\n&&{\\bm h}_k = \\bar{{\\bm A}}{\\bm h}_{k-1} + \\bar{{\\bm B}}{\\bm x}_k,\\\\\n&&{\\bm y}_k = {\\bm C}{\\bm h}_k + {\\bm D}{\\bm x}_k,\n\\end{eqnarray}'
new2 = '\\begin{equation}\n\\left\{\\begin{aligned}\n{\\bm h}_k &= \\bar{{\\bm A}}{\\bm h}_{k-1} + \\bar{{\\bm B}}{\\bm x}_k,\\\\\n{\\bm y}_k &= {\\bm C}{\\bm h}_k + {\\bm D}{\\bm x}_k.\n\\end{aligned}\\right.\n\\end{equation}'
if old2 in content:
    content = content.replace(old2, new2)
    print("2) SSM disc: DONE")
else:
    print("2) SSM disc: NOT FOUND")

# 3) a_n, b_n with \quad -> separate lines in aligned
old3 = '\\begin{eqnarray}\n\\bar{a}_n = \\exp(\\Delta \\cdot a_n), \\quad\n\\bar{b}_n = \\frac{\\exp(\\Delta \\cdot a_n) - 1}{a_n}.\n\\end{eqnarray}'
new3 = '\\begin{equation}\n\\left\{\\begin{aligned}\n\\bar{a}_n &= \\exp(\\Delta \\cdot a_n), \\\\\n\\bar{b}_n &= \\frac{\\exp(\\Delta \\cdot a_n) - 1}{a_n}.\n\\end{aligned}\\right.\n\\end{equation}'
if old3 in content:
    content = content.replace(old3, new3)
    print("3) a_n,b_n: DONE")
else:
    print("3) a_n,b_n: NOT FOUND")

# 4) K[t] with \quad trailing text -> separate into aligned
old4 = '\\begin{eqnarray}\nK[t] = \\sum_{n=1}^{N} C_n \\cdot \\bar{b}_n \\cdot \\bar{a}_n^t, \\quad t = 0, 1, \\ldots, T{-}1,\n\\end{eqnarray}'
new4 = '\\begin{equation}\nK[t] = \\sum_{n=1}^{N} C_n \\cdot \\bar{b}_n \\cdot \\bar{a}_n^t,\n\\end{equation}\n其中$K \\in \\mathbb{R}^T$为卷积核.'
if old4 in content:
    content = content.replace(old4, new4)
    print("4) K[t]: DONE")
else:
    print("4) K[t]: NOT FOUND")

# 5) J(a) cost function - long single line, split with \big[
old5 = '\\begin{eqnarray}\nJ({\\bm a}) = \\sum_{h=1}^{H}\\big[\\|\\hat{{\\bm s}}_{h} - {\\bm s}_{\\text{ref}}\\|_{{\\bm Q}}^2\n+ \\|{\\bm a}_{h-1}\\|_{{\\bm R}}^2\\big],\n\\end{eqnarray}'
new5 = '\\begin{equation}\n\\begin{aligned}\nJ({\\bm a}) &= \\sum_{h=1}^{H}\\Big[\\|\\hat{{\\bm s}}_{h} - {\\bm s}_{\\text{ref}}\\|_{{\\bm Q}}^2 \\\\\n&\\qquad + \\|{\\bm a}_{h-1}\\|_{{\\bm R}}^2\\Big].\n\\end{aligned}\n\\end{equation}'
if old5 in content:
    content = content.replace(old5, new5)
    print("5) J(a): DONE")
else:
    print("5) J(a): NOT FOUND")

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("\\nAll done!")
