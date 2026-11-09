# PRL 123, 116104 (2019) — 三幅远场偏振图

论文：*Circularly Polarized States Spawning from Bound States in the Continuum*，DOI 10.1103/PhysRevLett.123.116104。

本目录新增真实 Legume 本征模/辐射计算的三幅偏振图，未重新生成或覆盖既有能带图。复用根目录 `01_legume_Fig2_abc_TE_bands.ipynb` 的几何和 `../band_structure/` 的 Γ 模态数据。主图每种结构独立计算51×51点；每隔一点画一个椭圆以保证可读性，未用旋转/镜像复制数据。

## 模型与模态

- 正方晶格 a=450nm，薄膜厚100nm、n=2.02，上下空气。
- 孔面积固定247²nm²；L=sqrt(2S0/(2−α))，h=L，上底=(1−α)L。
- 原Notebook的上底朝+y，包围盒中心在(0,0)；α=1为三个非重复顶点。质心不等于包围盒中心，检查数据记录实际质心，未擅自平移/转动孔形。
- 沿用既有 Γ TE₂候选：α=0、0.1、1 的 f≈0.74905283、0.74882240、0.73643181。先以保存频率在原gmax=4.5处锚定，再通过带(guided_mode,G1,G2)标签的共有基底本征矢重叠追踪，跨gmax也使用重叠。二维扫描不是硬编码排序编号。
- Legume 1.0.3；gmax=4.5、abs截断、gmode_inds=[0,3]、numeig=12、kz_symmetry=None、guided-mode逐k精确计算。只对目标分支计算辐射。
- α=0扫描±0.07，α=0.1/1扫描±0.05（单位2π/a）；C点均在窗口内。

## 相干远场与相位核查

源码依据：`legume/gme/gme.py` 的 `_compute_rad_components`、`compute_rad_sp`、`ft_field_xy`，以及 `slab_modes.py:rad_modes`。Legume 已对 TE 耦合乘以−i以消除测试场额外相位。原始四组复系数全部保存；没有把强度比例当Jones矢量。

`rad_gvec` 实际识别 G=0，所有扫描点只有这个开放级。系数含密度态归一化，满足

    f_im = Im sqrt((2πf)² + i sum_channels |c|²)/(2π).

程序逐点核查。它们不是以V/m标定的电场，以下Jones只省略同一点共有的尺度/相位，S0为损耗归一化强度代理。

定义 p(k)=(kx,ky)/|k|，s(k)=(py,−px)，cosθ=sqrt(1−|k|²/(2πf)²)，本次正确映射是

    E_xy(k) = c_s(−k) s(k) − c_p(−k) cosθ p(k).

**α≠0不能直接套用先前C6结构同k的映射。** 使用互易对向的入射背景测试态与时间反演模态关系，耦合取−k；每个+k和−k都实际求解，`reciprocal_partner`记录索引。对背景薄膜的上方入射测试态，TE相位修正及TM入/出射电场的相对符号保留在上式中。不同k的本征矢整体相位任意，不影响各点Stokes参数。

独立核查 `validate.py:audit` 从实际模态的 D0−eps_avg E0 构造体极化，并用传输矩阵直接构造对向入射电场，进行24/48节点z积分，比较两个复数积分与(c_s(−k),−c_p(−k))，只拟合一个共同复因子。三个α、两个非镜面对称方向均验证，包括相位而非仅强度。同k系数在非对称结构的试算中残差可达0.23/0.94，不能使用。核查数字在validation/checks.json。

Γ的局部p/s基矢用求解器delta_gx=1e−15处理：α=0辐射低于阈值，方向掩膜；α≠0中心有辐射，保留其实际偏振。Stokes使用全部复数，S3=2Im(Ex*Ey)。ψ=atan2(S2,S1)/2，χ=asin(S3/S0)/2，椭圆短长轴比=|tanχ|。C点处主轴不唯一，但圆形本身不依赖该角度，C点的根由S1/S0=S2/S0=0求出。

颜色固定映射：S3/S0=−1蓝、0灰、+1红。按E(t)=Re[E exp(−iωt)]、从上方朝样品观察的约定，正S3为逆时针旋转，本文称LCP，负S3称RCP。未获取原论文PDF，尚不能确认原文观察方向及LCP/RCP命名是否与此相同；因此图中直接标注无歧义的S3/S0，不人为反转颜色。

## 运行

```bash
cd /workspace/legume
source .venv/bin/activate
export MPLBACKEND=Agg MPLCONFIGDIR=/workspace/setup-legume/matplotlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
P=reproduction_results/PRL_2019/polarization_maps/scripts
for a in 0 0.1 1; do
  python "$P/compute_farfield.py" --alpha "$a" --grid 11
  python "$P/compute_farfield.py" --alpha "$a" --grid 31
  python "$P/compute_farfield.py" --alpha "$a" --grid 51
done
python "$P/find_cpoints.py"
python "$P/validate.py"
python "$P/plot_maps.py"
python "$P/report.py"
```

`find_cpoints.py`从当前扫描圆偏振度极值自动选种子，逐次调用真实模态计算精化零点，检查非零S0；默认测试gmax=4.5/5.5/6.5/8.5。不是将图上预设位置作为计算输入。局部根的函数求值历史全部保存。

依赖环境使用已安装的NumPy1.26.4、SciPy1.14.1、Matplotlib3.9.4、autograd1.7.0；版本记录在validation/summary.json。无需COMSOL或外部服务。

## 数据与图像

- `data/alpha0.npz`、`alpha0p1.npz`、`alpha1.npz`：最终51×51完整数据，含kxy、全部候选频率、目标频率/虚频、Q、mode(0-based)、本征矢及基底标签、父点/重叠/间隙、四组复耦合、通道G矢量、Ex/Ey、全部Stokes、ψ/χ和掩膜。
- `data/*_grid11_*.npz`及`*_grid31_*.npz`：预扫描和网格检查；最终51网格只以alpha*.npz提交，避免重复。
- `data/*loops*.npz`：BIC和精化C点周围的真实闭合路径计算。
- `data/*_metadata.json`：每个最终扫描的参数。
- `validation/cpoints.json`：根坐标、圆偏振度、S0、残差及完整求根历史。
- `validation/checks.json`：几何面积/质心、独立互易积分和绕转数。
- `validation/validation_report.md`、`summary.json`：定量报告。
- `figures/alpha{0,0p1,1}_polarization.{png,pdf}`：三张独立图；`Fig2_polarization_comparison.{png,pdf}`：横向比较。PNG为300dpi，PDF为矢量。

`plot_maps.py`只读取保存数据，不调用求解器。检查点每48点落盘，参数签名防止错用；检查点和缓存不提交。已有能带文件全部保持原样。

## 比较范围与限制

已验证所述物理演化：α=0中心BIC偏振涡旋，α>0中心辐射出现并产生一对相反S3的C点，随α增大位置外移。原始TE₂只是既有能带候选，当前模态追踪、辐射和拓扑结果进一步验证其物理相容性；没有原论文数字化能带或PDF逐像素比对。梯形朝向沿用Notebook，尚未直接核对Fig.2(d)。

本模型为无色散、无吸收、无限周期薄膜，辐射使用GME微扰。C点位置存在有限gmax误差，报告保留所有截断下的独立结果；绘图标记使用与背景一致的gmax，绝不将高截断根移植到低截断图上冒充一致。Γ极高数值Q只表明BIC数值消光，不代表可信的有限实数Q。
