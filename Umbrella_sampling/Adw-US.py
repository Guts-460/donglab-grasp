# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
import pymbar
from scipy.special import logsumexp
import os

# 参数设置
nt = 50000
dx = 0.1
dt = 0.002
ksai = 0.1
Temp = 3
kB = 1.0
beta = 1.0 / (kB * Temp)

# 系统初始化
nx = 1000
x0 = 0
K = 1.0
xx = np.zeros(nx)
time = np.zeros(nt)
u = np.zeros(nx)

# PMF计算参数
x_min = 0.0
x_max = 12.0
nbins = 40
xbin = 0.3

# 伞形采样参数
K_umb0 = 5.0
nwin = 50
N_max = nt

# 数组初始化
x_win = np.linspace(0, 12.0, nwin)
K_umb = np.full(nwin, K_umb0)
N_k = np.full(nwin, nt, dtype=np.int32)

# 预分配数组
bin_center_i = np.linspace(x_min, x_max, nbins)
bin_kn = np.zeros((nwin, N_max), dtype=np.int32)
u_kn = np.zeros((nwin, N_max))
u_kln = np.zeros((nwin, nwin, N_max))

# 势能函数计算
for i in range(nx):
    xx[i] = x0 + i * dx
    x = x0 + i * dx
    u[i] = 0.1*K*(x-6)**4 - 2*(x-6)**2 + 0.5*(x-6)

# 时间序列
time = np.arange(nt) * dt

# 伞形采样模拟
xt = np.zeros((nwin, nt))
for iwin in range(nwin):
    xt[iwin, 0] = x_win[iwin] + random.uniform(-0.25, 0.25)
    
    for it in range(nt - 1):
        x_current = xt[iwin, it]
        fx = -0.4*K*(x_current-6.0)**3 + 4*(x_current-6.0) - 0.5
        fx_umb = -K_umb[iwin] * (x_current - x_win[iwin])
        fx_total = fx + fx_umb
        xt[iwin, it + 1] = x_current + fx_total / ksai * dt
        rnormal = random.gauss(0, 1)
        fB = np.sqrt(2 * Temp / ksai / dt) * rnormal
        xt[iwin, it + 1] += fB * dt

# 计算直方图分箱
delta = (x_max - x_min) / nbins
for k in range(nwin):
    for n in range(N_k[k]):
        bin_kn[k, n] = int((xt[k, n] - x_min) / delta)

# 计算势能矩阵
for k in range(nwin):
    for n in range(N_k[k]):
        u_kn[k, n] = 0.1*K*(xt[k, n] - 6)**4 - 2*(xt[k, n]-6)**2 + 0.5*(xt[k, n]-6)
        dx = xt[k, n] - x_win
        u_kln[k, :, n] = u_kn[k, n] + beta * K_umb * dx**2

# 使用MBAR计算PMF
try:
    mbar = pymbar.MBAR(u_kln, N_k, verbose=True)
    f_i, df_i = mbar.computePMF(u_kn, bin_kn, nbins)
except Exception as e:
    print(f"MBAR计算错误: {e}")
    hist, edges = np.histogram(xt.flatten(), bins=nbins, range=(x_min, x_max))
    f_i = -np.log(hist + 1e-10)
    f_i -= f_i.min()
    df_i = np.zeros_like(f_i)

# 保存数据到文件
data_to_save = np.column_stack((bin_center_i, f_i, df_i))
header = "position\t-logP\tdf"
np.savetxt('Adw-US.txt', data_to_save, 
           fmt='%.6f', 
           header=header, 
           delimiter='\t',
           comments='')

print("数据已保存为Adw-US.txt，格式为：")
print(header)
print("position\t-logP\tdf")
print(f"{bin_center_i[0]:.6f}\t{f_i[0]:.6f}")
print("...\t...")
print(f"{bin_center_i[-1]:.6f}\t{f_i[-1]:.6f}")

# 绘图设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制PMF
ax1.plot(bin_center_i, f_i, 'r-', linewidth=0.5, label='PMF')
ax1.fill_between(bin_center_i, f_i - df_i, f_i + df_i, color="#1CC1DF", alpha=0.6)
ax1.set_xlabel('Position (x)', fontsize=20)
ax1.set_ylabel('-logP', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_title('Potential of Mean Force', fontsize=20)
# ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=16)

# 绘制各窗口分布
for k in range(nwin):
    hist, edges = np.histogram(xt[k], bins=nbins, range=(x_min, x_max))
    prob = hist / hist.sum()
    ax2.plot(bin_center_i, prob, '-', linewidth=1.5, alpha=0.7)

ax2.set_xlabel('Position (x)', fontsize=20)
ax2.set_ylabel('Probability Density', fontsize=20)
ax2.set_title('Umbrella Sampling Distributions', fontsize=20)

plt.tight_layout()
plt.savefig('umbrella_sampling_results.tiff', dpi=300, bbox_inches='tight')
plt.show()