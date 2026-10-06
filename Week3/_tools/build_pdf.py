r"""把 content_theory.py / content_code.py 编译成 PDF（LaTeX + tectonic）。

与旧的 ReportLab 版本相比，改用真正的 LaTeX 排版后：
  * 矩阵用 \begin{bmatrix}，不需要手写渲染器；
  * 公式由 LaTeX 自动编号，可用 \label/\ref 交叉引用；
  * 图表由 LaTeX 自动编号，标题不再手工写“图 1”；
  * 中文交给 ctex/xeCJK，避头尾、标点、断行都由它处理。

依赖：tectonic（单文件发行版，按需下载宏包）。
运行：python _tools/build_pdf.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from content_code import CONTENT as CODE_CONTENT  # noqa: E402
from content_code import META as CODE_META  # noqa: E402
from content_theory import CONTENT as THEORY_CONTENT  # noqa: E402
from content_theory import META as THEORY_META  # noqa: E402

DOCUMENTS = [(THEORY_META, THEORY_CONTENT), (CODE_META, CODE_CONTENT)]

TECTONIC = Path.home() / "AppData/Local/Programs/tectonic/tectonic.exe"

# 正文字体缺字形时，改用等价的 LaTeX 数学写法。
# 初始集合来自之前对 SimSun 的覆盖检测（下标、上标、∇、减号等）；
# 编译后若有新的 Missing character 警告，再往这里补。
TEX_SYMBOLS = {
    # 带点/双点的导数写法（组合字符，字体里没有）
    "q⃛": r"$\dddot{q}$",
    "q̈": r"$\ddot{q}$",
    "θ̇": r"$\dot{\theta}$",
    "ẋ": r"$\dot{x}$",
    # 上标 / 下标
    "ᵀ": r"$^{\top}$",
    "⁻": r"$^{-}$",
    "¹": r"$^{1}$",
    "²": r"$^{2}$",
    "³": r"$^{3}$",
    "⁴": r"$^{4}$",
    "₀": r"$_{0}$",
    "₁": r"$_{1}$",
    "₂": r"$_{2}$",
    "₃": r"$_{3}$",
    "₅": r"$_{5}$",
    # 数学符号
    "∇": r"$\nabla$",
    "−": r"$-$",
    "⋯": r"$\cdots$",
    "∝": r"$\propto$",
    "∈": r"$\in$",
    "≡": r"$\equiv$",
    "↔": r"$\leftrightarrow$",
    "⇔": r"$\Leftrightarrow$",
    "→": r"$\rightarrow$",
    "′": r"$'$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "×": r"$\times$",
    "·": r"$\cdot$",
    "√": r"$\surd$",
    # 希腊字母（正文里应排成数学斜体）
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "ε": r"$\varepsilon$",
    "θ": r"$\theta$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "π": r"$\pi$",
    "ρ": r"$\rho$",
    "σ": r"$\sigma$",
    "τ": r"$\tau$",
    "φ": r"$\phi$",
    "ψ": r"$\psi$",
    "ω": r"$\omega$",
    "Δ": r"$\Delta$",
    "Ω": r"$\Omega$",
}

# 数学模式里的希腊字母（拼下标时需要 LaTeX 命令而不是 Unicode）
GREEK_TEX: dict[str, str] = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu",
    "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau",
    "φ": r"\phi", "ψ": r"\psi", "ω": r"\omega",
    "Δ": r"\Delta", "Ω": r"\Omega", "Ψ": r"\Psi", "Φ": r"\Phi",
}

# 下标是“词”而不是变量时排正体（v_max、T_LSPB 之类）
ROMAN_SUBS = {"max", "min", "peak", "new", "rand", "end", "old", "tot", "LSPB"}
# 这些名字是算子而不是变量，排正体
ROMAN_BASES = {"Rot", "Trans", "Var", "std"}
# 代码标识符：保持字面量，下划线照常转义
KEEP_LITERAL = {"dh_transform", "make_transform", "plan_with_fixed_duration",
                "fixed_duration", "rot_z_h", "transform", "action_space"}

MATH_TOKEN_RE = re.compile(
    r"[A-Za-z\u0370-\u03ff][A-Za-z0-9\u0370-\u03ff]*_[A-Za-z0-9\u0370-\u03ff]+")

# 少数记号直接写成想要的排版形式（自动规则拼不出这种嵌套下标）
MATH_SPECIAL = {
    "π_θold": r"$\pi_{\theta_{\mathrm{old}}}$",
    "π_θ": r"$\pi_{\theta}$",
}


def greek_run(s: str) -> str:
    """把一段文字里的希腊字母换成 LaTeX 命令，并用花括号隔开。

    不加花括号的话，`θ` + `old` 会拼成 `\thetaold`，LaTeX 会当成一个
    未定义命令而报错。
    """
    return "".join("{" + GREEK_TEX[ch] + "}" if ch in GREEK_TEX else ch for ch in s)

# 手写的“图 N”“表 N”前缀：交给 LaTeX 自动编号，这里去掉
MANUAL_LABEL_RE = re.compile(r"^(图|表)\s*\d+\s*[\u3000\s]*")


def strip_manual_label(caption: str) -> str:
    """去掉手写的“图 3　”前缀，避免和 LaTeX 自动编号重复。"""
    return MANUAL_LABEL_RE.sub("", caption)


def _convert_segment(segment: str) -> str:
    """在纯文本片段里把变量记号排成数学模式。"""
    for src, dst in MATH_SPECIAL.items():
        segment = segment.replace(src, dst)

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok in KEEP_LITERAL:
            return tok
        start, end = m.start(), m.end()
        if (start and segment[start - 1] == ".") or segment[end:end + 1] in ("_", "."):
            return tok
        if re.match(r"\.(py|png|csv|json|tex|pdf|log)\b", segment[end:end + 7]):
            return tok
        base, sub = tok.split("_", 1)
        base_tex = greek_run(base)
        sub_tex = greek_run(sub)
        if base in ROMAN_BASES:
            base_tex = r"\mathrm{" + base_tex + "}"
        if sub in ROMAN_SUBS:
            sub_tex = r"\mathrm{" + sub_tex + "}"
        return "$" + base_tex + "_{" + sub_tex + "}$"

    return MATH_TOKEN_RE.sub(repl, segment)


def texify_math_tokens(text: str) -> str:
    r"""把正文里的 v_max、r_t、T_AB 这类记号排成数学模式。

    利用上下文区分“变量下标”与“代码标识符”：紧跟 `_` 或 `.`、或前面是句点的
    （即属于文件名、模块路径、更长的蛇形命名）一律保持字面量。

    另外必须避开已经写在 $...$ 里的内容：内容文件里有的地方已经手写了 LaTeX
    数学（如 $p_0=T_{0n}p_n$），再转一次会把下标变成 \_ 从而报
    “Display math should end with $$”。
    """
    parts = text.split("$")
    return "$".join(_convert_segment(p) if i % 2 == 0 else p
                    for i, p in enumerate(parts))


# 很长的拉丁标识符（含 . 与 \_）在 CJK 段落里无法自动换行，容易溢出右边距
LONG_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\\_\-]{16,}")


def allow_breaks(text: str) -> str:
    """在长标识符的点号/下划线之后插 \\allowbreak，给它制造换行机会。"""
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        tok = tok.replace(".", ".\\allowbreak ")
        tok = tok.replace("\\_", "\\_\\allowbreak ")
        return tok

    parts = text.split("$")
    for i in range(0, len(parts), 2):      # 只处理数学模式之外的片段
        parts[i] = LONG_TOKEN_RE.sub(repl, parts[i])
    return "$".join(parts)


# 需要转义的 LaTeX 特殊字符（反斜杠与花括号由下方循环单独处理）
ESCAPES = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$",
           "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}


def tex_escape(text: str) -> str:
    """把正文里的特殊字符转义，并保留 <b> 粗体标记与数学模式。"""
    # 1) HTML 实体先换成占位符，免得后面和 <b> 标记混在一起
    text = text.replace("&lt;", "\x00LT\x00").replace("&gt;", "\x00GT\x00")
    text = text.replace("&amp;", "\x00AMP\x00")
    # 2) 粗体标记 -> \textbf{}
    text = re.sub(r"<b>(.*?)</b>", lambda m: r"\textbf{" + m.group(1) + "}", text, flags=re.S)
    # 3) 变量记号（v_max 之类）-> 数学模式；要放在符号替换之前，
    #    这样下标里的希腊字母才能拿到 LaTeX 命令
    text = texify_math_tokens(text)
    # 4) 其余数学符号 -> 数学模式（长序列先替换，避免被单字符规则拆开）
    for src, dst in TEX_SYMBOLS.items():
        text = text.replace(src, dst)
    # 5) 逐字符转义；命令名与 $...$ 内部原样保留
    out: list[str] = []
    i, in_math = 0, False
    while i < len(text):
        ch = text[i]
        if ch == "\\":                      # 一条命令：只吃掉命令名
            j = i + 1
            while j < len(text) and text[j].isalpha() and text[j].isascii():
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if ch == "$":                        # 切换数学模式
            in_math = not in_math
            out.append(ch)
            i += 1
            continue
        out.append(ch if in_math else ESCAPES.get(ch, ch))
        i += 1
    text = "".join(out)
    # 6) 还原占位符
    text = (text.replace("\x00LT\x00", r"$<$").replace("\x00GT\x00", r"$>$")
            .replace("\x00AMP\x00", r"\&"))
    # 7) 给很长的标识符（env.action\_space.sample()）插断点，避免行溢出
    return allow_breaks(text)


PREAMBLE = r"""
\documentclass[11pt,a4paper]{ctexart}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage[table]{xcolor}
\usepackage{listings}
\usepackage{caption}
\usepackage{geometry}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}

\geometry{left=2.5cm,right=2.5cm,top=2.4cm,bottom=2.6cm}
\linespread{1.42}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.25em}

\captionsetup{font=small,labelfont=bf,skip=5pt}
\captionsetup[table]{position=bottom,skip=4pt}

\definecolor{hdark}{HTML}{12304F}
\definecolor{hmid}{HTML}{1F4E79}
\definecolor{hlight}{HTML}{2F5D80}
\definecolor{thd}{HTML}{2F5D80}
\definecolor{tal}{HTML}{F2F5F8}

\newcommand{\hOne}[1]{\par\vspace{0.5em}%
  {\color{hdark}\Large\bfseries #1}\par\vspace{0.1em}%
  \noindent{\color{hdark}\rule{\linewidth}{0.9pt}}\par\vspace{0.35em}}
\newcommand{\hTwo}[1]{\par\vspace{0.8em}{\color{hmid}\large\bfseries #1}\par\vspace{0.25em}}
\newcommand{\hThree}[1]{\par\vspace{0.6em}{\color{hlight}\normalsize\bfseries #1}\par\vspace{0.15em}}

\lstset{
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  breakatwhitespace=false,
  frame=single,
  framesep=5pt,
  rulecolor=\color{gray!45},
  backgroundcolor=\color{gray!6},
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  xleftmargin=8pt,
  xrightmargin=8pt,
  aboveskip=0.7em,
  belowskip=0.7em,
}

\pagestyle{empty}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.25em}
"""

EPILOGUE = r"""
\end{document}
"""


def col_spec(widths: list[float]) -> str:
    """生成按比例分配宽度的 X 列（tabularx 会自动填满 \\textwidth）。

    不用 p{0.xx\\textwidth} 是因为那样算不入列间的 \\tabcolsep，
    列一多就会超出正文宽、压到右边距上。
    """
    n = len(widths)
    total = sum(widths) or 1.0
    return "".join(
        r">{\hsize=" + f"{n * w / total:.4f}" + r"\hsize\raggedright\arraybackslash}X"
        for w in widths
    )


def render_table(item: tuple) -> str:
    _, header, rows, widths, caption = item[:5]
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabularx}{\textwidth}{" + col_spec(widths) + "}",
        r"\toprule",
        r"\rowcolor{thd}" + " & ".join(
            r"\textcolor{white}{\textbf{" + tex_escape(c) + "}}" for c in header) + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i % 2 == 1:
            lines.append(r"\rowcolor{tal}")
        lines.append(" & ".join(tex_escape(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    if caption:
        lines.append(r"\caption{" + tex_escape(strip_manual_label(caption)) + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_math(latex: str) -> str:
    """去掉正文内容里手写的公式编号痕迹（LaTeX 会自动编号）。"""
    return latex


def convert_document(meta: dict, content: list) -> str:
    """把内容列表转成完整的 .tex 源码。"""
    body: list[str] = []
    bullets: list[str] = []

    def flush_bullets() -> None:
        if bullets:
            body.append(r"\begin{itemize}[leftmargin=1.6em,itemsep=0.15em,topsep=0.25em]")
            body.extend(r"\item " + b for b in bullets)
            body.append(r"\end{itemize}")
            bullets.clear()

    for item in content:
        kind = item[0]
        if kind != "b":
            flush_bullets()

        if kind == "title":
            # 居中环境自成一块：否则一旦删掉 subtitle，center 就闭合不了
            body.append(r"\begin{center}{\color{hdark}\Huge\bfseries "
                        + tex_escape(item[1]) + r"}\end{center}")
        elif kind == "subtitle":
            body.append(r"\begin{center}{\color{hmid}\large "
                        + tex_escape(item[1]) + r"}\end{center}")
        elif kind == "meta":
            body.append(r"\begin{center}{\color{hmid}\small "
                        + tex_escape(item[1]) + r"}\end{center}")
        elif kind == "gap":
            body.append(r"\vspace{" + f"{item[1]:.2f}" + "cm}")
        elif kind == "h1":
            body.append(r"\hOne{" + tex_escape(item[1]) + "}")
        elif kind == "h2":
            body.append(r"\hTwo{" + tex_escape(item[1]) + "}")
        elif kind == "h3":
            body.append(r"\hThree{" + tex_escape(item[1]) + "}")
        elif kind == "p":
            body.append(tex_escape(item[1]) + r"\par")
        elif kind == "pn":
            body.append(r"\noindent " + tex_escape(item[1]) + r"\par")
        elif kind == "b":
            bullets.append(tex_escape(item[1]))
        elif kind == "eq":
            body.append(r"\begin{equation}" + render_math(item[1]) + r"\end{equation}")
        elif kind == "mateq":
            prefix, rows = item[1], item[2]
            matrix = (r"\begin{bmatrix}" +
                      r" \\ ".join(" & ".join(cell for cell in row) for row in rows) +
                      r"\end{bmatrix}")
            head = (prefix + " ") if prefix else ""
            body.append(r"\begin{equation}" + head + matrix + r"\end{equation}")
        elif kind == "fig":
            path = (ROOT / "figures" / item[1]).as_posix()
            body.append(r"\begin{figure}[htbp]" + "\n" + r"\centering" + "\n"
                        + r"\includegraphics[width=\linewidth]{" + path + "}\n"
                        + r"\caption{" + tex_escape(strip_manual_label(item[2])) + "}\n"
                        + r"\end{figure}")
        elif kind == "table":
            body.append(render_table(item))
        elif kind == "code":
            body.append(r"\begin{lstlisting}" + "\n" + item[1] + "\n" + r"\end{lstlisting}")
        elif kind == "pagebreak":
            body.append(r"\clearpage")
        else:
            raise ValueError(f"未知内容类型: {kind}")

    flush_bullets()

    head = PREAMBLE + "\n".join([
        r"\title{" + tex_escape(meta["title"]) + "}",
        r"\author{MotrixLab 线上实习}",
        "",
    ])
    return head + r"\begin{document}" + "\n" + "\n\n".join(body) + "\n" + EPILOGUE


def build(meta: dict, content: list) -> tuple[Path, str, int]:
    """生成 .tex 并调用 tectonic 编译，返回 (PDF 路径, 编译输出, 退出码)。

    先把旧 PDF 删掉：否则一旦本次编译失败，上一次的旧文件还在，
    会被误判成“编译成功”。
    """
    tex_path = ROOT / (meta["filename"].removesuffix(".pdf") + ".tex")
    tex_path.write_text(convert_document(meta, content), encoding="utf-8")

    out_pdf = ROOT / meta["filename"]
    out_pdf.unlink(missing_ok=True)

    result = subprocess.run(
        [str(TECTONIC), "-X", "compile", str(tex_path), "--outdir", str(ROOT)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    log = (result.stdout or "") + (result.stderr or "")
    return out_pdf, log, result.returncode


def missing_characters(log: str) -> list[str]:
    """从 tectonic 输出里提取“缺字形”的字符。"""
    return sorted(set(re.findall(r"Missing character: There is no (.+?) in font", log)))


def main() -> None:
    status = 0
    all_missing: set[str] = set()

    for meta, content in DOCUMENTS:
        out_pdf, log, code = build(meta, content)

        if code != 0 or not out_pdf.exists():
            print(f"[失败] {meta['filename']}（tectonic 退出码 {code}）")
            for line in log.splitlines():
                if line.startswith("!") or "l." in line or "error" in line.lower():
                    print("   " + line.strip()[:160])
            status = 1
            continue

        size_kb = out_pdf.stat().st_size / 1024
        print(f"[pdf] {out_pdf.name}  ({size_kb:.0f} KB)")

        missing = missing_characters(log)
        if missing:
            all_missing.update(missing)
            print(f"      缺字形 {len(missing)} 个: {''.join(missing)}")

        overfull = log.count("Overfull \\hbox")
        if overfull:
            print(f"      排版溢出 {overfull} 处（不影响正确性，可后续微调）")

    if all_missing:
        print("\n把下面这些字符补进 TEX_SYMBOLS 再重新编译：")
        for ch in sorted(all_missing):
            print(f"    {ch!r}: r\"...\",")

    if status == 0:
        print("\n两份文档编译完成。")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
