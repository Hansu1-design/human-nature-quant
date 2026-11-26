# -*- coding: utf-8 -*-
"""
d131 嵌套 ZigZag：大段（20%）= 牛熊体制，小段（8%） = 体制过程中的波动
回答：牛市里和熊市里，这个两个月级别的波动各是什么样
"""
import numpy as np
import pandas as pd
from d130_probe_wave import load, zigzag

pd.set_option('display.width', 220)


def regime_segs(c, thr=0.20):
    """大阈值切出牛熊段，返回 [(i0, i1, dir, amp)]"""
    segs = zigzag(c, thr)
    # 合并：只保留方向确定的
    out = []
    for s in segs:
        i0, i1, dr, amp, _ = s
        if i1 - i0 < 5:
            continue
        out.append((i0, i1, dr, amp))
    return out


def main():
    d = load()
    c = d['close'].values
    n = len(c)

    # ---------- 大段 ----------
    big = regime_segs(c, 0.20)
    print('=' * 90)
    print(f'大段（20% 阈值）= 牛熊体制，共 {len(big)} 段')
    print('=' * 90)
    print(f'{"#":>3} {"起":>11} {"止":>11} {"方向":>4} {"幅度":>8} {"交易日":>6} {"月":>5}')
    for k, (i0, i1, dr, amp) in enumerate(big):
        print(f'{k:3d} {str(d.d.iloc[i0].date()):>11} {str(d.d.iloc[i1].date()):>11} '
              f'{"涨" if dr>0 else "跌":>4} {amp*100:+7.1f}% {i1-i0:6d} {(i1-i0)/20.3:5.1f}')

    # ---------- 嵌套：每个大段里的小波动 ----------
    print()
    print('=' * 90)
    print('嵌套分析：大段（牛熊）内部的小波动（8% 阈值）')
    print('=' * 90)

    rows = []
    for k, (i0, i1, dr, amp) in enumerate(big):
        sub = c[i0:i1 + 1]
        if len(sub) < 30:
            continue
        small = zigzag(sub, 0.08)
        sm = [s for s in small if s[1] - s[0] >= 3]
        if len(sm) < 2:
            continue
        # 只取与大段方向相反的小段 = 回调/反弹
        against = [s for s in sm if s[2] * dr < 0]
        along = [s for s in sm if s[2] * dr > 0]
        rows.append(dict(
            k=k, d0=d.d.iloc[i0].date(), d1=d.d.iloc[i1].date(),
            regime='牛' if dr > 0 else '熊', big_amp=amp,
            big_len=i1 - i0, n_small=len(sm),
            n_against=len(against), n_along=len(along),
            against_amp=np.median([abs(s[3]) for s in against]) if against else np.nan,
            against_len=np.median([s[1] - s[0] for s in against]) if against else np.nan,
            against_max=max([abs(s[3]) for s in against]) if against else np.nan,
            along_amp=np.median([abs(s[3]) for s in along]) if along else np.nan,
            along_len=np.median([s[1] - s[0] for s in along]) if along else np.nan,
        ))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    print()
    print('-' * 90)
    print('按体制汇总：')
    for reg in ['牛', '熊']:
        s = t[t.regime == reg]
        if len(s) == 0:
            continue
        print(f'\n【{reg}市】{len(s)} 段大行情')
        print(f'  大段幅度中位 {s.big_amp.median()*100:+.1f}%  时长中位 {s.big_len.median():.0f} 日 '
              f'({s.big_len.median()/20.3:.1f} 月)')
        print(f'  内部小波动（8%）个数中位 {s.n_small.median():.0f} 个 / 段')
        print(f'  其中"逆向"（牛里的回调 / 熊里的反弹）:')
        print(f'    个数中位 {s.n_against.median():.0f}，'
              f'幅度中位 {s.against_amp.median()*100:.1f}%，'
              f'最深 {s.against_max.max()*100:.1f}%，'
              f'时长中位 {s.against_len.median():.0f} 日 ({s.against_len.median()/20.3:.1f} 月)')
        print(f'  其中"顺向"（牛里的上涨 / 熊里的下跌）:')
        print(f'    个数中位 {s.n_along.median():.0f}，'
              f'幅度中位 {s.along_amp.median()*100:.1f}%，'
              f'时长中位 {s.along_len.median():.0f} 日 ({s.along_len.median()/20.3:.1f} 月)')

    print()
    print('-' * 90)
    print('关键不对称检验：牛市回调 vs 熊市反弹（逆向段）')
    nu = t[t.regime == '牛']
    be = t[t.regime == '熊']
    if len(nu) and len(be):
        print(f'  牛市回调幅度中位 {nu.against_amp.median()*100:.1f}%  '
              f'vs 熊市反弹幅度中位 {be.against_amp.median()*100:.1f}%   '
              f'差 {(nu.against_amp.median()-be.against_amp.median())*100:+.1f}pp')
        print(f'  牛市回调时长中位 {nu.against_len.median():.0f} 日  '
              f'vs 熊市反弹时长中位 {be.against_len.median():.0f} 日')
        print(f'  牛市里平均每段 {nu.n_against.mean():.1f} 次回调，'
              f'熊市里平均每段 {be.n_against.mean():.1f} 次反弹')

    # ---------- 用 40 日窗口在牛熊里看 ----------
    print()
    print('=' * 90)
    print('固定 40 日（2 个月）滚动收益，按当时体制拆开')
    print('=' * 90)
    # 体制标记：属于哪个大段
    reg_arr = np.full(n, 0, dtype=int)  # 1 牛 -1 熊
    for (i0, i1, dr, amp) in big:
        reg_arr[i0:i1 + 1] = dr
    w = 40
    rr = np.full(n, np.nan)
    rr[w:] = c[w:] / c[:-w] - 1
    for lab, val in [('牛市', 1), ('熊市', -1)]:
        m = (reg_arr == val) & ~np.isnan(rr)
        if m.sum() > 10:
            x = rr[m]
            print(f'{lab}: n={m.sum():5d}  均值 {x.mean()*100:+5.2f}%  sd {x.std()*100:5.2f}%  '
                  f'p5 {np.percentile(x,5)*100:+6.1f}%  p95 {np.percentile(x,95)*100:+6.1f}%  '
                  f'跌>8%占比 {(x<-0.08).mean()*100:4.1f}%  涨>8%占比 {(x>0.08).mean()*100:4.1f}%')

    # ---------- 每年/每体制 出现多少次逆向波动 ----------
    print()
    print('=' * 90)
    print('频率：牛市/熊市里，8% 逆向波动多久来一次')
    print('=' * 90)
    for reg, dr in [('牛', 1), ('熊', -1)]:
        all_against = []
        total_len = 0
        for (i0, i1, drr, amp) in big:
            if drr != dr:
                continue
            sub = c[i0:i1 + 1]
            if len(sub) < 30:
                continue
            small = zigzag(sub, 0.08)
            ag = [s for s in small if s[1] - s[0] >= 3 and s[2] * drr < 0]
            all_against += [abs(s[3]) for s in ag]
            total_len += len(sub)
        if all_against:
            print(f'{reg}市: 共 {total_len} 交易日 ({total_len/243:.1f} 年)，'
                  f'{len(all_against)} 次逆向波动 → 每 {total_len/max(len(all_against),1):.0f} 交易日'
                  f'（{total_len/max(len(all_against),1)/20.3:.1f} 个月）一次；'
                  f'幅度中位 {np.median(all_against)*100:.1f}%、'
                  f'p25 {np.percentile(all_against,25)*100:.1f}%、'
                  f'p75 {np.percentile(all_against,75)*100:.1f}%、'
                  f'max {max(all_against)*100:.1f}%')


if __name__ == '__main__':
    main()
