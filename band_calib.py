# -*- coding: utf-8 -*-
"""band_calib  §7.2 条件路径带 · 建仓校准卡（只打印，不进决策分支）

纪律边界（23 号 §7.2，2026-11-16 先生定）：
  ⑤ 唯一允许用法 = 建仓时打印三行日志；禁止把返回值接进任何 if / 仓位 / 出场判断。
  ③ 禁止句式「偏出带外 ⇒ 我判断错了 ⇒ 应该出场」。
  ① 引用四要素：档位 + 尺度 + 覆盖率 + n_eff，缺一即违规。

用法：
  from band_calib import card
  for ln in card(rA=1.043, when='2026-11-16'):
      print(ln)

  python band_calib.py            # 打印样例（当前最新 rA 位置）
  python band_calib.py 1.05       # 指定 rA

设计约束：本模块**不返回任何可用于判断的数值**，只返回字符串列表。
快照缺失时不抛异常（引擎不许因为一张日志表崩），只打印一行「跳过」。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, 'r63', 'band_snapshot.json')
CACHE = {}

ZONE_HI = 1.10


def load():
    if 's' not in CACHE:
        with open(SNAP, encoding='utf-8') as f:
            CACHE['s'] = json.load(f)
    return CACHE['s']


def _pct(x, sign=True):
    return (f'{x * 100:+.1f}%' if sign else f'{x * 100:.1f}%')


def card(rA, when=None):
    """返回 3 行字符串（或 1 行「不适用」）。调用方只许 print。"""
    try:
        s = load()
    except Exception:
        return ['[校准卡] 快照缺失或损坏，本轮跳过（不影响任何决策）']

    tag = f' [{when}]' if when else ''
    if rA is None or rA != rA:
        return [f'[校准卡]{tag} rA 无效，跳过']
    if rA > ZONE_HI:
        return [f'[校准卡]{tag} rA={rA:.3f} 不在出手区（>{ZONE_HI:.2f}）⇒ 本卡不适用；'
                f'§1 硬规则：不在 {ZONE_HI:.2f} 之上出手，此处不提供任何预期数字']

    band = {b['h']: b for b in s['band_A']}
    touch = {t['h']: t for t in s['touch_A']}
    cover = s['cover_A']
    b3, t3 = band['3月'], touch['3月']
    b1, b6 = band['1月'], band['6月']

    l1 = (f'[校准卡]{tag} rA={rA:.3f} → A 出手区（≤{ZONE_HI:.2f}，任意日口径）｜3 月：中位 {_pct(b3["p50"])}，'
          f'p5~p95 {_pct(b3["p05"])}~{_pct(b3["p95"])}｜'
          f'3 月内曾摸 +10% 的概率 {_pct(t3["up10"], False)}，曾见 −10% 的 {_pct(t3["dn10"], False)}')

    l2 = (f'[校准卡] 覆盖率（样本外，滚动估带）：1 月 {_pct(cover["1月"]["cover"], False)}'
          f'（n_eff {b1["n_eff"]}）／3 月 {_pct(cover["3月"]["cover"], False)}（n_eff {b3["n_eff"]}）；'
          f'6 月中位 {_pct(b6["p50"])} 但覆盖率只有 {_pct(cover["6月"]["cover"], False)} ⇒ 偏出是常态，不算看错')

    g = s['guards']
    l3 = (f'[校准卡] 这是历史条件频率，不是预测；只校准预期，不进任何决策分支；'
          f'累计偏离率 >{g["dev_rate_stop"]*100:.0f}% 即停用重算（§7.2 证伪线）')
    return [l1, l2, l3]


def main():
    import sys
    if len(sys.argv) > 1:
        rA = float(sys.argv[1])
        when = None
    else:
        import d141_eat_bite as D
        d = D.load()
        rA = float(d['rA'].iloc[-1])
        when = str(d['d'].iloc[-1].date())
    for ln in card(rA, when):
        print(ln)


if __name__ == '__main__':
    main()
