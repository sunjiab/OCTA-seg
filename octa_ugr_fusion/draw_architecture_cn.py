"""生成 UGR-Fusion 中文总图、模块详图和训练损失图。"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT_FILE = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
BLUE, TEAL, ORANGE, PURPLE = "#2464A0", "#158477", "#B66C12", "#7656A8"
INK, MUTED = "#17324A", "#536779"
FILLS = {BLUE: "#EDF4FC", TEAL: "#E9F7F2", ORANGE: "#FFF4E5", PURPLE: "#F3EEFA"}


class Canvas:
    def __init__(self, width, height, title, subtitle=""):
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width, height), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, width), ylim=(0, height))
        self.ax.axis("off")
        self.text(width / 2, height - .48, title, 23, INK, "bold")
        if subtitle:
            self.text(width / 2, height - .92, subtitle, 10.5, MUTED)

    def text(self, x, y, value, size=11, color=INK, weight="normal", ha="center", zorder=5):
        self.ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                     ha=ha, va="center", linespacing=1.45, zorder=zorder)

    def box(self, x, y, w, h, value, color=BLUE, size=11, fill=None, radius=.1, lw=1.6):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.035,rounding_size={radius}",
            facecolor=fill or FILLS.get(color, "white"), edgecolor=color, lw=lw, zorder=3))
        self.text(x + w / 2, y + h / 2, value, size, color)

    def arrow(self, points, color=BLUE, dashed=False, width=1.7, zorder=2):
        xs, ys = zip(*points)
        if len(points) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=color, lw=width,
                         ls="--" if dashed else "-", zorder=zorder)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
            color=color, lw=width, linestyle="--" if dashed else "-", zorder=zorder + 1))

    def circle(self, x, y, symbol, color=INK, radius=.28, size=18):
        self.ax.add_patch(Circle((x, y), radius, fc="white", ec=color, lw=1.7, zorder=4))
        self.text(x, y, symbol, size, color, zorder=6)

    def section(self, y, label):
        self.ax.plot([.45, self.width - .45], [y, y], color="#DCE5ED", lw=1.3)
        self.text(.55, y - .35, label, 13.5, INK, "bold", "left")

    def save(self, directory, stem):
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf", "svg"):
            self.fig.savefig(directory / f"{stem}.{suffix}", dpi=190, facecolor="white")
        self.fig.savefig(directory / f"{stem}_预览.jpg", dpi=70, facecolor="white")
        plt.close(self.fig)


def image_panel(canvas, array, x, y, w, label, color):
    panel = canvas.fig.add_axes([x / canvas.width, y / canvas.height,
                                 w / canvas.width, w / canvas.height])
    panel.imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    panel.set_xticks([]); panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    canvas.text(x + w / 2, y + w + .25, label, 13, color, "bold")
    canvas.text(x + w / 2, y - .22, "1 × 304 × 304", 9.5, color)


def draw_overall(images, case_id, output):
    c = Canvas(26, 15, "UGR-Fusion 总体结构：OCTA 主预测，OCT 选择性修正",
               f"真实输入：OCTA-500 病例 {case_id}，3 mm，ILM-OPL；图中省略 Batch 维")
    c.text(.55, 13.45, "① 两个模态分别提取多尺度特征", 14, INK, "bold", "left")
    image_panel(c, images[0], .55, 9.65, 2.45, "OCTA 原始投影", BLUE)
    image_panel(c, images[1], .55, 5.7, 2.45, "OCT 原始投影", TEAL)

    scales = [("A0", "32×304²"), ("A1", "64×152²"), ("A2", "128×76²"),
              ("A3", "256×38²"), ("A4", "512×19²")]
    oct_scales = [("O0", "8×304²"), ("O1", "16×152²"), ("O2", "32×76²"),
                  ("O3", "64×38²"), ("O4", "128×19²")]
    xs = [3.75, 5.5, 7.25, 9.0, 10.75]
    for i, (x, (name, shape), (oname, oshape)) in enumerate(zip(xs, scales, oct_scales)):
        h = 1.42 - i * .12
        c.box(x, 10.25 + i * .12, 1.35, h, f"{name}\n{shape}\n双卷积", BLUE, 9.5)
        c.box(x, 6.35 + i * .12, 1.35, h, f"{oname}\n{oshape}\n轻量卷积", TEAL, 9.5)
        if i:
            c.arrow([(xs[i - 1] + 1.35, 10.92), (x, 10.92)], BLUE)
            c.text(x - .2, 11.25, "池化÷2", 8.5, BLUE)
            c.arrow([(xs[i - 1] + 1.35, 7.02), (x, 7.02)], TEAL)
            c.text(x - .2, 7.35, "平均池化÷2", 8.3, TEAL)
    c.arrow([(3.0, 10.85), (3.75, 10.85)], BLUE)
    c.arrow([(3.0, 6.92), (3.75, 6.92)], TEAL)
    c.text(7.2, 12.25, "OCTA 编码器：强模态主干", 11.5, BLUE, "bold")
    c.text(7.2, 8.32, "OCT 辅助金字塔：8/16/32/64/128 通道", 11.5, TEAL, "bold")

    c.box(12.8, 10.1, 3.0, 1.75, "U-Net 解码器\n4次转置卷积上采样\n与 A3/A2/A1/A0 跳跃拼接", BLUE, 10.5)
    for i, x in enumerate(xs[:-1]):
        c.arrow([(x + .68, 11.7 + i * .03), (x + .68, 12.45 + i * .03),
                 (13.3 + i * .5, 12.45 + i * .03), (13.3 + i * .5, 11.85)], BLUE, True, 1.1)
    c.arrow([(12.1, 11.1), (12.8, 11.1)], BLUE)
    c.box(16.35, 10.25, 2.25, 1.45, "主预测 logits\nz_A\n1×304×304", BLUE, 11)
    c.arrow([(15.8, 10.98), (16.35, 10.98)], BLUE)

    c.box(16.1, 8.05, 3.0, 1.35, "主预测概率\np_A = sigmoid(z_A).detach()\n不确定性 u = 4p_A(1-p_A)", ORANGE, 10.5)
    c.arrow([(17.48, 10.25), (17.48, 9.4)], ORANGE, True)
    c.text(19.5, 8.72, "u≈1：概率接近0.5\nu≈0：概率接近0或1", 9, ORANGE, ha="left")

    c.text(12.85, 5.2, "② 五个尺度分别进行门控残差修正", 14, INK, "bold", "left")
    head_xs = [13.0, 14.7, 16.4, 18.1, 19.8]
    for i, hx in enumerate(head_xs):
        c.box(hx, 3.25, 1.35, 1.25, f"尺度{i}\n门控头\ng{i}×d{i}", PURPLE, 9.5)
        c.arrow([(xs[i] + .68, 10.25 + i * .12), (xs[i] + .68, 5.25),
                 (hx + .35, 5.25), (hx + .35, 4.5)], BLUE, False, 1.0)
        c.arrow([(xs[i] + .68, 6.35 + i * .12), (xs[i] + .68, 5.0),
                 (hx + 1.0, 5.0), (hx + 1.0, 4.5)], TEAL, False, 1.0)
        c.arrow([(17.0, 8.05), (17.0, 5.55), (hx + .68, 5.55), (hx + .68, 4.5)], ORANGE, True, .9)
    c.text(16.75, 2.92, "每个头内部见图2：输入 A_l、O_l、|A_l-O_l|、u_l，输出单通道 g_l×d_l", 9.5, MUTED)

    c.box(21.55, 3.25, 2.15, 1.25, "上采样到304²\n五尺度求平均\nr = mean(up(g×d))", TEAL, 9.5)
    for hx in head_xs:
        c.arrow([(hx + 1.35, 3.88), (21.55, 3.88)], TEAL, False, .85)
    c.box(21.35, 6.05, 2.6, 1.4, "有界不确定性修正\nq=0.1+0.9u\nδ=s·q·2·tanh(r/2)", ORANGE, 10)
    c.arrow([(22.62, 4.5), (22.62, 6.05)], TEAL)
    c.arrow([(19.1, 8.72), (22.65, 8.72), (22.65, 7.45)], ORANGE, True)

    c.circle(24.7, 10.95, "+", INK, .28, 18)
    c.arrow([(18.6, 10.98), (24.42, 10.98)], BLUE, False, 2.1)
    c.text(21.2, 11.28, "主预测直接保留", 10, BLUE)
    c.arrow([(23.95, 6.75), (24.7, 6.75), (24.7, 10.67)], ORANGE)
    c.box(23.55, 12.25, 2.0, 1.0, "最终 logits\nz=z_A+δ", PURPLE, 10.5)
    c.arrow([(24.7, 11.23), (24.7, 12.25)], PURPLE)
    c.box(20.55, 12.25, 2.25, 1.0, "Sigmoid + 阈值0.5\n最终血管掩膜", BLUE, 10)
    c.arrow([(23.55, 12.75), (22.8, 12.75)], BLUE)

    c.text(.6, 1.65, "关键保护机制", 12, INK, "bold", "left")
    c.box(.6, .45, 5.4, .95, "残差输出层零初始化\n训练开始时 δ=0，因此最终输出精确等于 OCTA 主预测", BLUE, 10)
    c.box(6.35, .45, 5.4, .95, "修正幅度受限\n|δ|≤2；训练前5轮 s 从0.2线性增加到1", ORANGE, 10)
    c.box(12.1, .45, 5.4, .95, "不确定性不是错误真值\nq最低为0.1，仍允许修改过度自信的错误", ORANGE, 10)
    c.box(17.85, .45, 7.55, .95, "蓝色=OCTA主线；绿色=OCT辅助；橙色=不确定性；紫色=融合\n真实标签只参与训练损失，不进入推理 forward", PURPLE, 9.5)
    c.save(output, "UGR_图1_总体结构_中文")


def draw_head(output):
    c = Canvas(22, 12, "UGR-Fusion 图2：单尺度门控残差头内部",
               "该模块在304²、152²、76²、38²、19²五个尺度重复；以下用第 l 个尺度表示")
    c.text(.6, 10.65, "输入与通道对齐", 14, INK, "bold", "left")
    c.box(.7, 8.55, 3.3, 1.15, "OCTA特征 A_l\nC_A × H_l × W_l", BLUE, 11)
    c.box(.7, 6.75, 3.3, 1.15, "OCT特征 O_l\nC_O × H_l × W_l", TEAL, 11)
    c.box(.7, 4.95, 3.3, 1.15, "不确定性图 u\n1 × 304 × 304", ORANGE, 11)
    c.box(4.75, 8.55, 2.75, 1.15, "1×1卷积投影\nC_A → 16", BLUE, 11)
    c.box(4.75, 6.75, 2.75, 1.15, "1×1卷积投影\nC_O → 16", TEAL, 11)
    c.box(4.75, 4.95, 2.75, 1.15, "双线性缩放\n304² → H_l×W_l", ORANGE, 10.5)
    c.arrow([(4, 9.12), (4.75, 9.12)], BLUE)
    c.arrow([(4, 7.32), (4.75, 7.32)], TEAL)
    c.arrow([(4, 5.52), (4.75, 5.52)], ORANGE, True)

    c.box(8.35, 7.75, 2.35, 1.15, "逐元素差异\n|A-O|\n16通道", PURPLE, 10.5)
    c.arrow([(7.5, 9.12), (8.0, 9.12), (8.0, 8.55), (8.35, 8.55)], BLUE)
    c.arrow([(7.5, 7.32), (8.0, 7.32), (8.0, 8.1), (8.35, 8.1)], TEAL)
    c.box(11.55, 6.65, 3.3, 2.25, "通道拼接\n[A, O, |A-O|, u_l]\n16+16+16+1\n= 49通道", PURPLE, 11.5)
    c.arrow([(7.5, 9.12), (11.2, 9.12), (11.2, 8.45), (11.55, 8.45)], BLUE)
    c.arrow([(7.5, 7.32), (11.55, 7.32)], TEAL)
    c.arrow([(10.7, 8.32), (11.55, 8.05)], PURPLE)
    c.arrow([(7.5, 5.52), (11.2, 5.52), (11.2, 6.85), (11.55, 6.85)], ORANGE, True)

    c.box(15.75, 6.65, 2.9, 2.25, "特征混合\n3×3卷积：49→16\nGroupNorm(4组)\nSiLU激活", BLUE, 11)
    c.arrow([(14.85, 7.78), (15.75, 7.78)], PURPLE)
    c.box(19.25, 8.25, 2.1, 1.4, "门控分支\n1×1卷积\nSigmoid→g_l∈(0,1)", ORANGE, 10.5)
    c.box(19.25, 5.05, 2.1, 1.4, "残差分支\n1×1卷积\n零初始化→d_l", TEAL, 10.5)
    c.arrow([(18.65, 7.78), (18.95, 7.78), (18.95, 8.95), (19.25, 8.95)], ORANGE)
    c.arrow([(18.95, 7.78), (18.95, 5.75), (19.25, 5.75)], TEAL)
    c.circle(18.1, 3.65, "×", PURPLE, .32, 20)
    c.arrow([(20.3, 8.25), (20.3, 4.45), (18.1, 4.45), (18.1, 3.97)], ORANGE)
    c.arrow([(20.3, 5.05), (20.3, 3.65), (18.42, 3.65)], TEAL)
    c.box(14.0, 2.85, 3.05, 1.55, "该尺度输出\nR_l = g_l × d_l\n1 × H_l × W_l", PURPLE, 11.5)
    c.arrow([(17.78, 3.65), (17.05, 3.65)], PURPLE)
    c.box(9.35, 2.85, 3.35, 1.55, "统一空间尺寸\n双线性上采样到304²\nup(R_l)", TEAL, 11)
    c.arrow([(14, 3.62), (12.7, 3.62)], TEAL)
    c.box(4.5, 2.85, 3.55, 1.55, "五尺度聚合\nr = 1/5 · Σ up(R_l)\n单通道有符号修正", TEAL, 11)
    c.arrow([(9.35, 3.62), (8.05, 3.62)], TEAL)

    c.section(2.25, "每个量的作用")
    c.box(.75, .45, 4.7, 1.15, "A 与 O\n保留两模态各自的信息，不直接相加", BLUE, 10)
    c.box(5.75, .45, 4.7, 1.15, "|A-O|\n显式表示两模态在当前尺度的差异", PURPLE, 10)
    c.box(10.75, .45, 4.7, 1.15, "g_l\n控制该位置是否允许辅助分支发挥作用", ORANGE, 10)
    c.box(15.75, .45, 5.5, 1.15, "d_l\n正值提高血管logit，负值降低血管logit\n零初始化保证训练初始不扰动主预测", TEAL, 9.7)
    c.save(output, "UGR_图2_门控残差头_中文")


def draw_training(output):
    c = Canvas(22, 12, "UGR-Fusion 图3：训练监督、损失组成与模型选择",
               "推理时不需要标签；标签仅在训练和评估阶段使用")
    c.text(.6, 10.65, "前向输出", 14, INK, "bold", "left")
    c.box(.75, 8.5, 3.2, 1.4, "配对输入\nOCTA + OCT", PURPLE, 11)
    c.box(4.75, 8.5, 3.6, 1.4, "UGR-Fusion forward\n不接收真实标签", PURPLE, 11)
    c.arrow([(3.95, 9.2), (4.75, 9.2)], PURPLE)
    c.box(9.2, 9.15, 3.1, 1.15, "OCTA主输出\nanchor logits z_A", BLUE, 11)
    c.box(9.2, 7.45, 3.1, 1.15, "融合最终输出\nfinal logits z", TEAL, 11)
    c.arrow([(8.35, 9.2), (9.2, 9.72)], BLUE)
    c.arrow([(8.35, 9.2), (9.2, 8.02)], TEAL)
    c.box(.75, 6.15, 3.2, 1.4, "真实标签 y\n二值血管掩膜", ORANGE, 11)
    c.text(.75, 5.7, "训练专用", 9.5, ORANGE, "bold", "left")

    c.text(13.2, 10.65, "四项损失", 14, INK, "bold", "left")
    c.box(13.2, 9.25, 3.55, 1.05, "L_seg = BCE(z,y)+Dice(z,y)\n最终分割主损失", TEAL, 10)
    c.box(17.45, 9.25, 3.55, 1.05, "L_aux = BCE(z_A,y)+Dice(z_A,y)\n保持OCTA主分支可独立预测", BLUE, 9.7)
    c.box(13.2, 7.45, 3.55, 1.05, "L_keep\n只惩罚正确前景概率被融合压低", ORANGE, 9.8)
    c.box(17.45, 7.45, 3.55, 1.05, "L_clDice（可选）\n软骨架连通性损失，默认权重0", PURPLE, 9.7)
    c.arrow([(12.3, 8.02), (12.85, 8.02), (12.85, 9.78), (13.2, 9.78)], TEAL)
    c.arrow([(12.3, 9.72), (17.45, 9.78)], BLUE)
    c.arrow([(3.95, 6.85), (12.7, 6.85), (12.7, 9.52), (13.2, 9.52)], ORANGE, True, 1.1)
    c.arrow([(3.95, 6.85), (17.1, 6.85), (17.1, 9.52), (17.45, 9.52)], ORANGE, True, 1.1)
    c.arrow([(3.95, 6.85), (12.75, 6.85), (12.75, 7.98), (13.2, 7.98)], ORANGE, True, 1.1)
    c.arrow([(3.95, 6.85), (17.1, 6.85), (17.1, 7.98), (17.45, 7.98)], ORANGE, True, 1.1)
    c.arrow([(12.3, 9.72), (12.65, 9.72), (12.65, 7.72), (13.2, 7.72)], BLUE, True)
    c.arrow([(12.3, 8.02), (17.15, 8.02), (17.15, 7.72), (17.45, 7.72)], TEAL, True)

    c.box(7.15, 4.5, 7.7, 1.2,
          "总损失 L = L_seg + 0.3·L_aux + 0.1·L_keep + λ_cl·L_clDice\n默认 λ_cl=0；开启示例：--cldice-weight 0.1",
          PURPLE, 11.5)
    for x in (14.98, 19.22):
        c.arrow([(x, 7.45), (x, 6.25), (11.0, 6.25), (11.0, 5.7)], PURPLE, False, 1.1)
    for x in (14.98, 19.22):
        c.arrow([(x, 9.25), (x, 6.4), (11.0, 6.4), (11.0, 5.7)], PURPLE, False, 1.1)

    c.section(3.85, "L_keep 的具体计算")
    c.box(.75, 1.95, 4.45, 1.05, "候选像素集合\nM = y · I[sigmoid(z_A) ≥ 0.5]", ORANGE, 10.5)
    c.box(6.0, 1.95, 4.8, 1.05, "仅计算概率下降\nD = ReLU(sigmoid(z_A).detach() - sigmoid(z))", ORANGE, 9.8)
    c.box(11.6, 1.95, 3.9, 1.05, "保留损失\nL_keep = sum(M·D) / max(sum(M),1)", ORANGE, 9.8)
    c.arrow([(5.2, 2.48), (6.0, 2.48)], ORANGE)
    c.arrow([(10.8, 2.48), (11.6, 2.48)], ORANGE)
    c.box(16.3, 1.65, 4.85, 1.65,
          "作用与边界\n只保护标签为前景且主分支已判对的像素\n不保护anchor假阳性；属于软约束，不保证Recall上升",
          ORANGE, 9.5)
    c.arrow([(15.5, 2.48), (16.3, 2.48)], ORANGE)

    c.text(.75, .95, "训练与选择：", 11, INK, "bold", "left")
    c.text(3.0, .95, "AdamW → 验证集micro Dice选择best.pt → 连续20轮未提升则早停 → 方案冻结后仅一次正式测试", 10.2, MUTED, ha="left")
    c.text(.75, .48, "注意：不同辅助损失权重会改变总loss尺度；比较方法时主要看相同协议下的Dice、IoU、Precision、Recall及逐病例差值。", 9.5, MUTED, ha="left")
    c.save(output, "UGR_图3_训练损失_中文")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="10451")
    parser.add_argument("--output-dir", type=Path, default=HERE / "assets/中文结构图")
    args = parser.parse_args()
    if not FONT_FILE.is_file():
        raise FileNotFoundError(FONT_FILE)
    font_manager.fontManager.addfont(FONT_FILE)
    family = font_manager.FontProperties(fname=FONT_FILE).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 11,
                         "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    root = ROOT / "data/OCTA-500数据集/3mm"
    images = []
    for modality in ("OCTA", "OCT"):
        path = root / f"{modality}(ILM_OPL)" / f"{args.case_id}.bmp"
        with Image.open(path) as source:
            images.append(np.asarray(source.convert("L")).copy())
    draw_overall(images, args.case_id, args.output_dir)
    draw_head(args.output_dir)
    draw_training(args.output_dir)
    print(f"已生成到：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
