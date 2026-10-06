"""统一的绘图设置：中文字体 + 无界面后端。

所有复现脚本共用。用法：

    from mpl_style import setup
    plt = setup()
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 按优先级挑一个系统里存在的中文字体
CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "PingFang SC", "WenQuanYi Zen Hei", "SimSun"]


def pick_cjk_font() -> str | None:
    """返回系统里第一个可用的中文字体名，找不到则返回 None。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in CJK_CANDIDATES if name in available), None)


def setup(box_figsize: bool = False):
    """配置 matplotlib 并返回 pyplot 模块。"""
    picked = pick_cjk_font()
    plt.rcParams["font.sans-serif"] = ([picked] if picked else []) + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免缺字形
    if box_figsize:
        plt.rcParams["figure.figsize"] = (7.0, 4.2)
    if picked is None:
        print("[warn] 未找到中文字体，图中中文可能显示为方框")
    return plt
