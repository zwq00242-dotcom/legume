import json,sys,platform,importlib.metadata as im
import numpy as np
from compute_farfield import ROOT

def main():
    cp=json.loads((ROOT/'validation/cpoints.json').read_text());checks=json.loads((ROOT/'validation/checks.json').read_text());stats=[]
    for a,tag in zip([0,.1,1],['alpha0','alpha0p1','alpha1']):
        d=np.load(ROOT/'data'/f'{tag}.npz');assert len(d['kxy'])==51**2
        old=np.load(ROOT/'data'/f'{tag}_grid31_g4.5.npz')
        common=[]
        for i,k in enumerate(old['kxy']):
            j=np.argmin(np.linalg.norm(d['kxy']-k,axis=1))
            if np.linalg.norm(d['kxy'][j]-k)<1e-12 and d['valid'][j] and old['valid'][i]:common.append((i,j))
        ii,jj=np.array(common).T
        delta=np.max(abs(d['normalized_stokes'][jj]-old['normalized_stokes'][ii]))
        mirror=np.array([np.argmin(np.linalg.norm(d['kxy']-[-k[0],k[1]],axis=1)) for k in d['kxy']]);rev=d['reciprocal_partner'];ok=d['valid'] & d['valid'][mirror]
        merr=np.max(abs(d['normalized_stokes'][ok,2]+d['normalized_stokes'][mirror[ok],2]))
        assert delta<1e-7 and merr<1e-7
        assert np.all(d['nopen_u']==1) and np.all(d['nopen_l']==1)
        assert d['overlap'].min()>.9
        np.testing.assert_allclose(d['S0']**2,d['S1']**2+d['S2']**2+d['S3']**2,rtol=1e-12,atol=1e-25)
        center=np.argmin(np.linalg.norm(d['kxy'],axis=1))
        if a==0:assert not d['valid'][center]
        else:assert d['valid'][center]
        stats.append(dict(alpha=a,points=len(d['kxy']),valid=int(sum(d['valid'])),min_overlap=float(d['overlap'].min()),min_gap=float(d['gap'].min()),max_abs_S3=float(np.nanmax(abs(d['normalized_stokes'][:,2]))),gamma_Q=float(d['Q'][center]),gamma_S0=float(d['S0'][center]),grid31_vs51_stokes_error=float(delta),mirror_x_S3_error=float(merr),TR_frequency_error=float(max(abs(d['frequency']-d['frequency'][rev]))),TR_Q_relative_error=float(max(abs(d['Q'][d['valid']]/d['Q'][rev[d['valid']]]-1)))))
    summary=dict(scan=stats,environment=dict(python=sys.version,platform=platform.platform(),packages={p:im.version(p) for p in ['legume-gme','numpy','scipy','matplotlib','autograd']}),max_reciprocity_residual=max(r['relative_residual'] for r in checks['reciprocity']))
    (ROOT/'validation/summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# PRL 2019 偏振场验证报告','','三幅图均来自真实 Legume 模态；每幅最终数据为51×51，绘图每隔一点取样以避免椭圆重叠。未生成任何新能带图。','','## 模式与二维扫描','']
    for x in stats:lines.append(f"- α={x['alpha']}: {x['points']}点，有效偏振{x['valid']}点，最小追踪重叠{x['min_overlap']:.8f}，最小频率间隙{x['min_gap']:.6g}；31/51公共点Stokes误差{x['grid31_vs51_stokes_error']:.3e}。Γ S0={x['gamma_S0']:.6g}、Q={x['gamma_Q']:.6g}。")
    lines+=['','## C点局部求根与截断收敛','','坐标单位均为2π/a。每次函数求值均重新计算本征模和复辐射，非插值或预设位置。表中S3符号直接按2Im(Ex*Ey)计算。','','| α | gmax | kx | ky | S3/S0 | 零点残差 |','|---:|---:|---:|---:|---:|---:|']
    for r in cp:lines.append(f"|{r['alpha']}|{r['gmax']}|{r['kxy'][0]:.10f}|{r['kxy'][1]:.10g}|{r['normalized_stokes'][2]:.9f}|{r['residual']:.3e}|")
    lines += ['', 'gmax=4.5→8.5的|kx|变化约12.3%(α=0.1)和4.0%(α=1)；6.5→8.5仍约4.0%和1.3%。这些坐标是有限截断估计，不宣称已达到任意精度。']
    lines+=['','## 绕转数与Q检查','']
    for r in checks['loops']:lines.append('- '+str(r))
    lines+=['','## 远场、对称性与限制','',f"独立体极化互易积分复数比值的最大相对残差{summary['max_reciprocity_residual']:.3e}。必须用-k通道；同k转换在非对称孔形会给出错误旋向。",'', 'S3镜面对称与时间反演频率/Q数值误差详见summary.json。不能要求破坏C2后的场满足原有C2旋向关系。', '', 'gmax=4.5的C点位置还存在有限截断误差；表列更高gmax的重新求根结果，未偷偷将高精度C点标记到低精度背景上。所有主图的C点标记使用与背景一致的gmax=4.5结果。', '', '沿用原Notebook的上底朝+y几何，未翻转坐标。多边形的包围盒中心在原点，质心随α偏移，面积均为247²nm²；三角形只有三个非重复顶点。', '', '未取得论文PDF，因此没有直接核对Fig.2(d)朝向、坐标窗口和论文LCP/RCP观察定义。图中红/蓝始终代表实际S3正/负；在本报告e^(-iωt)、从上方朝样品观察的约定下，分别标记LCP/RCP。与原论文的手性命名仍需原图确认。', '', '模型使用n=2.02无色散、无限周期薄膜、GME微扰辐射。原TE2对应关系继承已有候选频率并以重叠连续追踪，不声称完成论文定量数字化拟合。']
    (ROOT/'validation/validation_report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
