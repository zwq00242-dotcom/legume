# 辐射振幅与偏振重建约定

本计算使用仓库中的 Legume 1.0.3 源码，未修改求解器。核查位置：

- `legume/gme/gme.py`: `rad_coup`, `rad_gvec`, `_compute_rad_components`, `compute_rad_sp`, `ft_field_xy`。
- `legume/gme/slab_modes.py`: `rad_modes` 与 TE/TM 传输矩阵。
- `legume/gme/matrix_elements.py`: `mat_te_te`, `mat_te_tm`, `mat_tm_te`, `mat_tm_tm`。
- 仓库官方示例 `docs/examples/14_GME_polarization_mixing.ipynb`。

## 系数不是强度

`compute_rad_sp(ik,[mode])` 返回五项：总、s、p 的虚频率，以及复数通道字典和对应 G 矢量字典。它不会自动把本次结果写进 `gme.rad_coup`/`freqs_im`；这些属性通常由 `run_im` 填充。本计算直接保存函数返回值。每次仅计算追踪的目标模态。

源码对 TE 的矩阵元乘以 `-1j`，注释明确说明这是消除 TE 场原有的额外 i，使 TE/TM 电场同相。不能再额外添加任意相位。四个键 `u_te,u_tm,l_te,l_tm` 都保留了相干的复振幅。

令 Ω=2πf，C² 为上下包层、所有开放衍射级及两种偏振的 |c|² 之和，则代码计算

    f_im = Im sqrt(Ω² + i C²) / (2π).

因此 c 是损耗归一化振幅，不是 SI 单位的电场。密度态因子 sqrt(πρ) 已包含在 c 中；同一个 G 通道的 s、p 因子相同。程序逐点重建这一关系并检查一致性。Jones 矢量只省略同一 k 点上两分量共有的尺度/相位；Stokes 方向、归一化参数与拓扑荷不依赖该因子。S0 不能解释为已校准的远场辐照度。

## 固定坐标中的电场

Legume 的局部面内基矢为

    p=(kx,ky)/|k|,
    s=(py,-px).

使用源码的 `delta_gx=1e-15` 与求解器保持一致；在 Γ 点不赋予偏振。零级通道通过实际返回的 `rad_gvec` 找出，绝不假定它总是第一个通道。所有本次采样点只有 G=0 开放，包括四角。

`rad_modes` 的返回系数在辐射矩阵元调用中以 `(As,Bs)=(Ys,Xs)` 传入。在上方包层，对应来自上方的背景散射测试态。`ft_field_xy` 的约定给出 TE 测试电场沿 s；TM 的入射与出射面内电场相对其 H 基矢具有不同符号。结合对向 (-k) 的互易测试态，必须保留这个相对符号。

本结构、本 C2 对称且关于薄膜中面对称的分支，复数远场在全局 Cartesian 坐标中为（相差一个共有因子）：

    (Ex,Ey) = u_te*s - u_tm*cos(theta)*p,
    cos(theta)=sqrt(1-|k|²/Ω²).

这不是把 s/p 分别当作 x/y，也不是从目标绕转数选择符号。`cos(theta)` 是出射 p 偏振的面内投影。全局电场不做旋转、镜像或角度修正。该映射针对这里的结构/分支验证，不宣称是适用于任意不对称结构的无条件通用公式。

这里不对这两个系数作整体复共轭。共轭虽不改变偏振主轴，却会改变 S3 符号。我们通过以下独立的数值互易积分，核实相对符号及椭圆率约定。任意本征矢整体相位仅使两个 Jones 分量同时乘以同一相位，不影响偏振。

## 独立的体极化互易积分

`validate_bic.py:reciprocal_audit` 不从 `rad_coup` 生成测试场：它调用背景薄膜的传输矩阵系数，解析构造 -k 处入射场，在薄膜中使用 Gauss-Legendre 积分。实际模态的体极化零阶 Fourier 分量从完整模态场独立计算：

    P0(z)=D0(z)-eps_avg*E0(z).

这是 (eps-eps_avg)E 的 G=0 分量；D 与 E 来自 `ft_field_xy`，后者明确使用逆介电矩阵进行卷积。对向入射测试场归一化到同一个全局电场轴：s=q，p_out=(cos(theta)*p,-sin(theta))。测试场 TM 电场必须除以薄膜 eps_avg，这与以 H 归一化的 TM 辐射态一致。

程序计算两个复数积分 Is=∫E_test,s·P0 dz、Ip=∫E_test,p·P0 dz，并比较 `(Is,Ip)` 与 `(u_te,-u_tm)`，只允许拟合一个共同复数因子。没有给 s/p 分别拟合因子或更改符号。比较 3 个非对称方向 k 点与 24/48 个 z 积分节点，数值及残差保存在 `source_audit.json`。这验证的是复数 Jones 比值，包含相对相位；不是单纯核查强度或绕转数。

Γ 点的 GME 微扰损耗约 1e-30，不是对实际有限结构 Q 的可靠定量预测。图中明确掩膜中心；不使用该点的任意局部基矢来赋予偏振方向。有限远离中心的损耗和拓扑路径另行计算。
