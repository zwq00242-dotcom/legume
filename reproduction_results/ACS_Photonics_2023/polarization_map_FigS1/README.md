# ACS Photonics 2023 — Fig. S1(a) 的 Legume 数值复现

论文：*Realizing Tunable Evolution of Bound States in the Continuum and Circularly Polarized Points by Symmetry Breaking*, DOI: 10.1021/acsphotonics.2c01522。

主图是 **961 个独立 k 点的真实 GME 本征模和复数辐射耦合计算**，没有使用解析涡旋、旋转复制、坐标反射或人为角度修正。Γ 点是未定义偏振的掩膜点，图上仅用黑点标识。所有非中心点均保留原始复数场。

## 复用与目标模态

此前 ACS 结果位于远端 `work` 分支，当前分支从初始 master 快进至 `a540db8` 后复用；未覆盖已有能带文件。结构数值直接读取 `../band_structure/highres/metadata.json`，与 `examples/paper_reproduction/acs2023_fig1a_highres.py:build_structure` 完全一致。

三角晶格 `Lattice('hexagonal')`，a=724 nm，h=100 nm，r=200 nm，eps=4，上下空气。晶格矢量 a1=(1/2,sqrt(3)/2)，a2=(1/2,-sqrt(3)/2)；b1=(2π,2π/sqrt(3))，b2=(2π,-2π/sqrt(3))。坐标始终为该全局右手 Cartesian x/y，输入 Legume 的 k=2π*(图上坐标)。

原程序已将 Γ 点 f=0.8874621514、λ=815.809439 nm 的非简并分支作为红色目标带。本次只在这个已存频率处初始化一次，再按带有 (guided_mode,G_index1,G_index2) 标签的共有基底本征矢重叠，在按半径排序的二维点集中从最近已追踪点继续。跨 gmax 使用该原始 Γ 模态的重叠选择对应分支。不会为了获得目标拓扑荷重新选择能带。保存了父点、选择重叠、第二候选差值、最近频率间隙和目标本征矢，供审计。所用扫描中目标恰好始终是排序第2带，这是追踪结果，不是扫描中的硬编码选择。

原有 Γ 波长仅是仓库中的候选匹配，而非原论文数字化拟合。补充 PDF 不在仓库；尝试访问 ACS Figshare 和 ReadTheDocs 被本环境网络策略拒绝（CONNECT 403）。因此已验证给定结构、既有候选分支与目标高阶 BIC 的相容性，**没有宣称与 PDF 逐像素或绝对角度比对完成**。

## 运行

从仓库根目录运行，使用已配置的 `.venv`：

```bash
cd /workspace/legume
source .venv/bin/activate
export MPLBACKEND=Agg MPLCONFIGDIR=/workspace/setup-legume/matplotlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/scripts/compute_modes.py --grid 11
python reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/scripts/compute_modes.py --grid 31
python reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/scripts/validate_bic.py
python reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/scripts/plot_polarization.py
python reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/scripts/report_results.py
```

依赖：Legume 1.0.3（本仓库可编辑安装），NumPy 1.26.4，SciPy 1.14.1，Matplotlib 3.9.4，contourpy 1.3.1，autograd 1.7.0。完整环境版本在 `data/calculation_metadata.json`。安装可用 `python -m pip install -e '.[test,autodiff]' 'numpy==1.26.4' 'scipy==1.14.1' 'matplotlib==3.9.4' 'contourpy==1.3.1' 'autograd==1.7.0'`。

主扫描 gmax=4.5、abs 截断、gmode_inds=[0,3]、numeig=12，guided-mode 逐 k 精确求解，`kz_symmetry=None`。先计算本征模，只对追踪的目标调用 `compute_rad_sp`。保存原始上下 s/p 系数、实际 G=0 通道、所有候选频率及目标模态信息。每48点保存本地检查点，可断点续算；检查点和缓存不提交 Git，完整汇总 `.npz` 会提交。改变参数须使用新的计算名称，程序检查检查点参数，避免错误复用。

## 文件

- `figures/FigS1a_polarization_map.png`：300 dpi 主图；同名 PDF 为矢量图。
- `figures/FigS1b_Q_factor.png`：辐射 Q 辅助图，Γ 数值地板点掩膜。
- `figures/mode_tracking_diagnostic.png`：波长、重叠、S3/S0 和邻带间隙。
- `data/grid31.npz`：完整二维数值结果，数组按 `meshgrid(x,y)` 后行优先排列。
- `data/farfield_complex.npz`：kx/ky、实/虚频、Q、复 s/p、Ex/Ey、G 通道及损耗和。
- `data/polarization_stokes.npz`：S0…S3、归一化 Stokes、psi、椭圆率角、掩膜与阈值。
- `data/mode_tracking.npz`：所有频率、目标序号（**0-based**）、父点、重叠/差值、间隙、目标本征矢与基底标签。
- `data/grid11.npz`：真实预扫描；`data/loops_*.npz`：独立闭合路径的原始计算结果。
- `validation/validation_report.md`、`convergence_results.csv`、`source_audit.json`：物理验证及可审计数字。
- `validation/source_conventions.md`：s/p 相位、归一化、Cartesian 转换和互易积分推导。

图像只从已保存的数据重建，`plot_polarization.py` 不调用求解器。偏振主轴 psi=atan2(S2,S1)/2，S3=2 Im(Ex* Ey)，不丢弃虚部。S0 使用损耗归一化单位，不是绝对辐照度。以相对最大 S0 的 1e-12 和绝对 1e-28 两者较大者作为掩膜阈值，另显式排除 Γ。掩膜不会把零振幅变成指定方向。

## 结果与限制

定量结果由 `validation/validation_report.md` 给出。主图复现了给定 C6 结构的中心高阶 BIC 与 -2 偏振绕转。中心损耗落入浮点/微扰方法数值地板，不能将约1e29的 Q 当作可靠有限实数测量。材料采用无色散 n=2，结构无限周期、无吸收，GME 辐射是微扰估计；有限尺寸、材料损耗和 COMSOL 全波共振极点不在本计算范围。较大 gmax 和扩展 guided-mode 基底用于检验拓扑稳定性，不能将拓扑稳定等同于所有频率、Q 已达到任意精度。
