import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import pandas as pd

def potential_energy(x, y):
    """计算势能面"""
    A1 = 5.0
    a1, b1, c1 = 0.3, 0.5, 0.2
    x1, y1 = 1.0, 2.0

    A2 = 4.5
    a2, b2, c2 = 0.4, 0.3, -0.3
    x2, y2 = 4.0, 3.5

    B = 3.0
    x0, y0 = 3, 3.0
    sigma_x, sigma_y = 2.0, 1.5

    well1 = -A1 * np.exp(-a1*(x-x1)**2 - b1*(y-y1)**2 - c1*(x-x1)*(y-y1))
    well2 = -A2 * np.exp(-a2*(x-x2)**2 - b2*(y-y2)**2 - c2*(x-x2)*(y-y2))
    barrier = B * np.exp(-(x-x0)**4/sigma_x**4 - (y-y0)**4/sigma_y**4)

    return well1 + well2 + barrier

def plot_sampled_path(Nstep, d, N):
    """绘制势能面并添加采样路径"""
    # 创建网格
    x = np.linspace(0, 8, 500)
    y = np.linspace(0, 6, 500)
    X, Y = np.meshgrid(x, y)
    Z = potential_energy(X, Y)

    # 读取采样路径数据
    path_data = pd.read_csv(f'results/path_sampled_{d}_{N}d.txt', sep='\t')
    x_coords = path_data['x'].values
    y_coords = path_data['y'].values
    steps = path_data['step'].values
    
    n_points = min(Nstep, len(x_coords))
    x_coords = x_coords[:n_points]
    y_coords = y_coords[:n_points]
    steps = steps[:n_points]

    # 图形创建
    os.makedirs('figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    # 势能面 (亮色调，固定范围)
    contour = ax.contourf(
        X, Y, Z,
        levels=15,
        cmap="Blues_r",
        alpha=0.9,
        vmin=-4, vmax=4
    )

    # colorbar
    cbar = fig.colorbar(contour, ax=ax, pad=0.05)
    cbar.set_label('Potential Energy', fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks([-5, -4, -3, -2, -1, 0, 1])

    # 路径颜色：白 → 黑 渐变
    cmap_bw = colors.LinearSegmentedColormap.from_list("white_to_black", ["white", "black"])
    norm = plt.Normalize(vmin=1, vmax=n_points)

    for i in range(n_points):
        ax.scatter(x_coords[i], y_coords[i],
                   color=cmap_bw(norm(i+1)), s=25, alpha=1, edgecolors="black")

    # 起点终点
    ax.scatter(0.5, 2, c="white", s=60, marker='^', label='t=0', edgecolors="#08CDF4")
    ax.scatter(x_coords[-1], y_coords[-1], c="black", s=60, marker='v', label=f't={n_points}', edgecolors="#F408F0")

    # 坐标与标题
    ax.set_xlabel('X', fontsize=28)
    ax.set_ylabel('Y', fontsize=28)
    ax.legend(fontsize=24, loc='lower right', framealpha=0)
    ax.tick_params(axis='both', which='major', labelsize=24)

    # === 新增：鼠标坐标显示 ===
    coord_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', horizontalalignment='right',
                         fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    def on_mouse_move(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            coord_text.set_text(f'X: {x:.3f}\nY: {y:.3f}')
        else:
            coord_text.set_text('')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    # ==========================

    # 保存图像
    plt.tight_layout()
    plt.savefig(
        f'figures/2d_path_sampled_{d}_{N}d.tiff',
        format='tiff', dpi=300, bbox_inches='tight'
    )
    plt.show()

if __name__ == "__main__":
    Nstep = 28
    dstep = 6   # 步长×10
    dir_N = 32  # 方向数
    plot_sampled_path(Nstep, dstep, dir_N)