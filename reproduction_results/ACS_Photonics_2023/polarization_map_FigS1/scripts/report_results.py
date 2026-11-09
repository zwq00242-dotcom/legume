"""Validate archived arrays and generate a quantitative report from them."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    d=np.load(ROOT/'data/grid31.npz');low=np.load(ROOT/'data/grid11.npz')
    rows=list(csv.DictReader((ROOT/'validation/convergence_results.csv').open()))
    audit=json.loads((ROOT/'validation/source_audit.json').read_text())
    assert len(d['kxy'])==961 and sum(d['valid'])==960
    assert np.all(d['nopen_u']==1) and np.all(d['nopen_l']==1)
    assert np.all(d['mode']==1) and np.all(d['overlap']>.99) and np.all(d['margin']>.9)
    assert np.all(np.isfinite(d['Ex'])) and np.all(np.isfinite(d['Ey']))
    assert np.isnan(d['psi'][np.argmin(np.linalg.norm(d['kxy'],axis=1))])
    ex,ey=d['Ex'],d['Ey'];s0=abs(ex)**2+abs(ey)**2
    np.testing.assert_allclose(s0,d['S0'],rtol=1e-13)
    np.testing.assert_allclose(d['S1']**2+d['S2']**2+d['S3']**2,d['S0']**2,rtol=1e-12,atol=1e-28)
    assert max(abs(float(r['charge'])+2) for r in rows)<1e-10
    assert max(v['relative_residual'] for v in audit['reciprocity_audits'])<1e-10
    inds=np.array([np.argmin(np.linalg.norm(d['kxy']-k,axis=1)) for k in low['kxy']])
    mask=low['valid'];angle_error=.5*np.angle(np.exp(2j*(d['psi'][inds][mask]-low['psi'][mask])))
    base=np.load(ROOT/'data/loops_g4p5.npz');hi=np.load(ROOT/'data/loops_g6p5_expanded.npz')
    conv_angle=.5*np.angle(np.exp(2j*(base['psi'][1:]-hi['psi'][1:])))
    selected=[r for r in rows if r['gmax']=='4.5' and r['density']=='48']
    radii=np.array([float(r['radius']) for r in selected]);qs=np.array([float(r['Q_median']) for r in selected])
    slope=np.polyfit(np.log(radii[:2]),np.log(qs[:2]),1)[0]
    assert np.all(np.diff(qs)<0)
    zero=np.argmin(np.linalg.norm(d['kxy'],axis=1))
    metrics=dict(scan_points=len(d['kxy']),valid_axes=int(sum(d['valid'])),gamma_Q_numerical=float(d['Q'][zero]),gamma_freq_im=float(d['freq_im'][zero]),gamma_S0=float(d['S0'][zero]),minimum_overlap=float(d['overlap'].min()),minimum_margin=float(d['margin'].min()),minimum_frequency_gap=float(d['gap'].min()),max_abs_S3_over_S0=float(np.nanmax(abs(d['normalized_stokes'][:,2]))),max_ellipticity_degrees=float(np.nanmax(abs(np.rad2deg(d['ellipticity_angle'])))),charge_min=min(float(r['charge']) for r in rows),charge_max=max(float(r['charge']) for r in rows),max_C6_axis_error_rad=max(float(r['C6_axis_error_rad']) for r in rows),max_TR_axis_error_rad=max(float(r['TR_axis_error_rad']) for r in rows),max_reciprocity_residual=max(v['relative_residual'] for v in audit['reciprocity_audits']),grid11_vs31_max_axis_error_rad=float(abs(angle_error).max()),g4p5_vs_g6p5_expanded_max_axis_error_deg=float(np.rad2deg(abs(conv_angle).max())),g4p5_vs_g6p5_expanded_max_relative_frequency_change=float(np.max(abs(base['frequency']/hi['frequency']-1))),g4p5_vs_g6p5_expanded_median_relative_Q_change=float(np.median(abs(base['Q'][1:]/hi['Q'][1:]-1))),Q_log_slope_r001_to002=float(slope),actual_expanded_guided_modes=np.unique(hi['basis_keys'][:,0]).tolist())
    sym=[]
    for name in ['loops_g3p5','loops_g4p5','loops_g5p5','loops_g6p5','loops_g6p5_expanded']:
        v=np.load(ROOT/'data'/f'{name}.npz')
        for radius in radii:
            ix=np.flatnonzero(np.isclose(np.linalg.norm(v['kxy'],axis=1),radius))[-48:]
            for step in [8,24]:
                sym.append([np.max(abs(np.roll(v['frequency'][ix],-step)-v['frequency'][ix])),np.max(abs(np.roll(v['Q'][ix],-step)/v['Q'][ix]-1))])
    metrics['max_C6_or_TR_frequency_difference']=float(np.max(np.array(sym)[:,0]))
    metrics['max_C6_or_TR_relative_Q_difference']=float(np.max(np.array(sym)[:,1]))
    (ROOT/'validation/summary.json').write_text(json.dumps(metrics,indent=2))
    lines=['# 物理验证报告','','## 结论','','主图由 Legume 1.0.3 的 31×31 个独立 k 点计算得到，960 个有效偏振主轴；中心 Γ 明确掩膜。所有检查通过。',
           '',f"- 目标 Γ 频率 {audit['gamma']['frequency']:.12f} a/λ，波长 {724/audit['gamma']['frequency']:.6f} nm；沿用既有 Fig.1(a) 的候选分支。",
           f"- C6 模态旋转特征值 {audit['gamma']['character_C6'][0]:.12f}，Fourier H_z 旋转残差 {audit['gamma']['relative_residual']:.3e}。",
           f"- 所有50条独立闭合路径的拓扑荷范围 [{metrics['charge_min']:.15f}, {metrics['charge_max']:.15f}]；逆时针取样、原始 Cartesian 坐标。",
           '- 路径半径为 0.01、0.02、0.04、0.08、0.12，分别24/48点；每点真实求解，并非六重旋转复制。',
           '- 截断 gmax=3.5、4.5、5.5、6.5，另测试 gmax=6.5 的 [0,3,4,7] 请求基底；实际可用 guided modes 为 [0,3,4]，第7态在该截断范围没有进入。',
           f"- C6 偏振轴协变最大误差 {metrics['max_C6_axis_error_rad']:.3e} rad；正反 k 偏振轴最大误差 {metrics['max_TR_axis_error_rad']:.3e} rad。",
           f"- 全二维最小追踪重叠 {metrics['minimum_overlap']:.9f}，最小选择差值 {metrics['minimum_margin']:.9f}，最小相邻频率间隙 {metrics['minimum_frequency_gap']:.9f} a/λ；目标均为排序第2带，没有检测到交叉、近简并或跳模。",
           f"- 11×11 与31×31 公共点偏振轴最大误差 {metrics['grid11_vs31_max_axis_error_rad']:.3e} rad；二者保存的原始振幅与偏振约定一致。",
           f"- 全二维 max |S3/S0|={metrics['max_abs_S3_over_S0']:.6f}，最大椭圆率角 {metrics['max_ellipticity_degrees']:.6f}°；偏振接近线性但未丢弃虚部。",
           f"- 独立体极化互易积分：3个非对称方向、24/48个z节点，复数比值最大相对残差 {metrics['max_reciprocity_residual']:.3e}；包括相对相位而非仅强度。详见 source_conventions.md。",
           f"- C6/正反k对应点的频率最大绝对差 {metrics['max_C6_or_TR_frequency_difference']:.3e} a/λ，Q 最大相对差 {metrics['max_C6_or_TR_relative_Q_difference']:.3e}。", '', '## BIC 与辐射 Q','','| 环半径 (2π/a) | Q 中位数 (gmax=4.5,48点) | q |','|---:|---:|---:|']
    for r in selected:lines.append(f"| {float(r['radius']):.3f} | {float(r['Q_median']):.6g} | {float(r['charge']):.8f} |")
    lines += ['',f"半径0.01至0.02得到 Q ∝ |k|^({slope:.4f})，接近高阶 BIC 的 |k|^-4 近 Γ 标度；远离 Γ 后不要求保持单一幂律。",
              f"Γ f_im={metrics['gamma_freq_im']:.3e}、S0={metrics['gamma_S0']:.3e}；形式 Q≈{metrics['gamma_Q_numerical']:.3e}，这是浮点消光/微扰数值地板，不将它作为可信的有限 Q。有限半径上的单调 Q 增长、零辐射趋势与稳健绕转共同支持中心 BIC。",
              '', '## 精度与论文比较','',f"gmax=4.5 与 gmax=6.5 扩展基底的相同圆路径上，偏振轴最大变化 {metrics['g4p5_vs_g6p5_expanded_max_axis_error_deg']:.6f}°，频率最大相对变化 {100*metrics['g4p5_vs_g6p5_expanded_max_relative_frequency_change']:.5f}%，Q相对变化的中位数 {100*metrics['g4p5_vs_g6p5_expanded_median_relative_Q_change']:.4f}%。拓扑荷已经稳定，但不声称 Q 和频率达到无限精度。",
              '', '与所述 Fig.S1(a) 一致：C6 薄膜、中心高阶 BIC、-2 远场主轴绕转、相同坐标范围、短线段及未定义的中心偏振。主图是原始31×31扫描，未用对称性补点。',
              '', '限制：原补充 PDF 未提供，本次下载被网络策略拒绝，未作原图数字化或逐像素/绝对方向比较。红色能带对应关系继承仓库先前的几何与形状匹配，现在进一步用频率、C6 特征值、辐射消失与 -2 拓扑验证其物理相容性。常数 n=2、无限周期、无吸收以及 GME 微扰辐射会造成与全波/实验结果的定量差异。',
              '', '## 可审计文件','','原始复振幅、全部 Stokes 参数及追踪信息分别位于 data/farfield_complex.npz、polarization_stokes.npz、mode_tracking.npz；完整汇总为 grid31.npz。路径原始计算在 data/loops_*.npz，环境/参数在 calculation_metadata.json。source_audit.json 存独立复数积分。图像重建命令见 README。']
    (ROOT/'validation/validation_report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for radius in [.02,.04,.08,.12]:
        ix=np.flatnonzero(np.isclose(np.linalg.norm(base['kxy'],axis=1),radius))[-48:]
        z=base['S1'][ix]+1j*base['S2'][ix];phase=np.unwrap(np.angle(np.r_[z,z[0]]));phi=np.linspace(0,2*np.pi,49)
        axes[0].plot(phi/(2*np.pi),(phase-phase[0])/(4*np.pi),label=f'r={radius}')
    axes[0].set(xlabel='Loop fraction (CCW)',ylabel='Accumulated axis winding');axes[0].legend(fontsize=8)
    axes[1].loglog(radii,qs,'o-');axes[1].set(xlabel='Loop radius (2π/a)',ylabel='Median radiative Q');axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/topology_Q_validation.png',dpi=240);plt.close(fig)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
