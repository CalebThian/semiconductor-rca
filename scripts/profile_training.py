# -*- coding: utf-8 -*-
"""
profile_training.py  ——  量 MT-NGC 訓練的時間花在哪

為什麼需要先量
--------------
  你的 mt_neural_granger.py 是「每個 (task, target) 一個獨立 MLP」的結構
  （self.predictors[task][i]，make_dataset(..., target_var)，檔頭註明
  "This is the V1 baseline; backbone sharing is a future ablation."）。

  N 個參數就是 N 個獨立訓練，乘上 task 數。要知道能加速多少，得先分清
  時間是花在「數學」還是「Python 與 DataLoader 的往返」—— 兩者的修法完全
  不同，猜錯會白做。

這支程式回答三個問題
--------------------
  1. 實際訓練了幾個 task？其中幾個是你真正要用的？
     （若 output_graph 只取 task2b，卻訓練了三個 task，就有 2/3 的時間
       是完全浪費的 —— 這是風險最低、效益最高的一刀。）
  2. 單一 target 的訓練要多久？乘上 target 數是否解釋得了總時間？
  3. 同樣的形狀下，「逐一訓練」對「全 target 批次訓練」差幾倍？
     （這一項不需要你的模型碼，用等價形狀的合成資料直接量。）

執行
----
  在專案根目錄：
    python -u profile_training.py --csv data\\raw\\semiconductor\\case1.csv

  只跑第 3 項（不需要專案模組，純形狀測試）：
    python -u profile_training.py --shape-only --n 300 --rows 800

環境：Python 3.6+ / numpy / torch
"""

from __future__ import print_function

import argparse
import os
import sys
import time


def say(s=''):
    print(s, flush=True)


def hr(t):
    say('')
    say('-' * 64)
    say(t)
    say('-' * 64)


# ------------------------------------------------------------------ 第 3 項

def shape_benchmark(N, T, L, H, bs, epochs, threads):
    """同形狀下，逐一訓練 vs 全 target 批次訓練。不需要專案模組。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    if threads > 0:
        torch.set_num_threads(threads)

    def per_target():
        X = torch.randn(T, N * L)
        Y = torch.randn(T, N)
        t0 = time.time()
        for i in range(N):
            net = nn.Sequential(nn.Linear(N * L, H), nn.ReLU(),
                                nn.Linear(H, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, Y[:, i:i + 1]),
                batch_size=bs, shuffle=True)
            for _ in range(epochs):
                for xb, yb in dl:
                    opt.zero_grad()
                    l = (((net(xb) - yb) ** 2).mean()
                         + 1e-2 * net[0].weight.abs().sum())
                    l.backward()
                    opt.step()
        return time.time() - t0

    class Batched(nn.Module):
        """所有 target 共用同一個輸入，所以第一層可以合成單一矩陣乘：
        (B, N*L) @ (N*L, N*H)。權重 view 成 (N*L, N, H) 之後，
        每個 target 的輸入重要度與 group lasso 都沿 target 維索引，
        與逐一訓練在數學上等價。"""

        def __init__(s):
            super(Batched, s).__init__()
            s.N, s.H = N, H
            s.W1 = nn.Parameter(torch.randn(N * L, N * H)
                                * (1.0 / (N * L) ** 0.5))
            s.b1 = nn.Parameter(torch.zeros(N * H))
            s.W2 = nn.Parameter(torch.randn(N, H) * (1.0 / H ** 0.5))
            s.b2 = nn.Parameter(torch.zeros(N))

        def forward(s, x):
            h = torch.addmm(s.b1, x, s.W1).relu().view(-1, s.N, s.H)
            return torch.einsum('bnh,nh->bn', h, s.W2) + s.b2

    def batched():
        X = torch.randn(T, N * L)
        Y = torch.randn(T, N)
        net = Batched()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        t0 = time.time()
        for _ in range(epochs):
            perm = torch.randperm(T)
            for k in range(0, T, bs):
                i = perm[k:k + bs]
                opt.zero_grad()
                l = (((net(X[i]) - Y[i]) ** 2).mean()
                     + 1e-2 * net.W1.abs().sum())
                l.backward()
                opt.step()
        return time.time() - t0

    return per_target(), batched()


# ------------------------------------------------------------------ 主流程

def main():
    p = argparse.ArgumentParser(description='MT-NGC 訓練耗時剖析')
    p.add_argument('--csv', default=None, help='原始資料（跑第 1、2 項用）')
    p.add_argument('--task', default='task2b',
                   help='你實際使用的 output_graph')
    p.add_argument('--lookback', type=int, default=3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=2,
                   help='形狀測試用的 epoch 數（只影響測量，不是你的設定）')
    p.add_argument('--threads', type=int, default=0,
                   help='限制 torch 執行緒數（0 = 不限制）')
    p.add_argument('--shape-only', action='store_true',
                   help='只做第 3 項，不匯入專案模組')
    p.add_argument('--n', type=int, default=None,
                   help='shape-only 時的參數數')
    p.add_argument('--rows', type=int, default=800,
                   help='shape-only 時的列數')
    args = p.parse_args()

    say('=' * 64)
    say('MT-NGC 訓練耗時剖析')
    say('=' * 64)

    N, T = args.n, args.rows

    if not args.shape_only:
        sys.path.insert(0, os.getcwd())
        hr('[1] 訓練了幾個 task？')
        try:
            from src.models.mt_neural_granger import TASKS
            say('程式中定義的 TASKS：%s（共 %d 個）'
                % (list(TASKS), len(TASKS)))
            say('你實際使用的：%s' % args.task)
            if args.task in TASKS and len(TASKS) > 1:
                say('')
                say('⚠️ 若 train_mt_neural_granger 對每個 task 都訓練一輪，')
                say('   則有 %d/%d 的訓練時間產出你不使用的圖。'
                    % (len(TASKS) - 1, len(TASKS)))
                say('   請確認 train_mt_neural_granger 裡是否有')
                say('   「for task in TASKS」這樣的迴圈 —— 若有，加一個只訓練')
                say('   指定 task 的參數，是風險最低的 %.1fx 加速。'
                    % (len(TASKS) / 1.0))
        except Exception as e:
            say('無法匯入 TASKS（%s）' % e)
            say('請手動確認：train_mt_neural_granger 是否對所有 task 都訓練。')

        hr('[2] 單一 target 的訓練成本')
        try:
            from scripts.residual_anomaly import split_by_symbol
            if not args.csv or not os.path.exists(args.csv):
                say('未給 --csv 或找不到檔案，跳過本項。')
            else:
                normal, _ab, names = split_by_symbol(args.csv)
                N = len(names)
                T = len(normal)
                say('Normal %d 列／參數 %d 個' % (T, N))
                say('')
                say('每個 target 的第一層權重：(%d × %d) × %d = %s'
                    % (N, args.lookback, args.hidden,
                       format(N * args.lookback * args.hidden, ',')))
                say('乘上 %d 個 target：%s 個參數'
                    % (N, format(N * N * args.lookback * args.hidden, ',')))
                say('→ 參數量隨 N 平方成長，這是總成本的主項。')
        except Exception as e:
            say('無法取得資料（%s），跳過本項。' % e)

    if N is None:
        N = 100
        say('')
        say('（未取得實際參數數，形狀測試用 N=%d）' % N)

    hr('[3] 逐一訓練 vs 全 target 批次訓練（同形狀）')
    say('N=%d, 列數=%d, lookback=%d, hidden=%d, batch=%d, epochs=%d'
        % (N, T, args.lookback, args.hidden, args.batch_size, args.epochs))
    say('（測量中，逐一訓練那邊會比較久……）')
    _bench = None
    try:
        import torch
        say('torch %s，執行緒 %d' % (torch.__version__,
                                     torch.get_num_threads()))
        # 暖機，排除第一次呼叫的初始化成本
        shape_benchmark(8, 200, args.lookback, 16, 64, 1, args.threads)
        a, b = shape_benchmark(N, T, args.lookback, args.hidden,
                               args.batch_size, args.epochs, args.threads)
        _bench = (a, b)
        say('')
        say('  逐一訓練：%.2f 秒（%.2f 秒/epoch）' % (a, a / args.epochs))
        say('  批次訓練：%.2f 秒（%.2f 秒/epoch）' % (b, b / args.epochs))
        say('  加速：%.1fx' % (a / max(b, 1e-9)))
        say('')
        if a / max(b, 1e-9) > 5:
            say('→ 加速明顯，表示時間主要花在 Python 與 DataLoader 的往返，')
            say('  不是數學。把 target 批次化就能拿到這個倍數。')
        else:
            say('→ 加速有限，表示在這個 N 下已經是計算受限（compute-bound）。')
            say('  批次化只能去掉常數開銷；要再快就得減少計算量本身：')
            say('  參數量隨 N 平方成長，N 砍半＝計算量降為 1/4。')
            say('  用 reduce_params.py 把 --topk 再調小，效益大於任何工程優化。')
    except ImportError:
        say('沒有 torch，跳過本項。')

    hr('結論與建議順序')
    say('1. 先確認是否訓練了用不到的 task —— 零模型變更，最高 3x')
    say('2. 降低 N —— 計算量隨 N² 成長，N 300→150 就是 4 倍')
    say('3. 把 target 批次化 —— N 小時 10~20x，N 大時約 2x')
    say('4. 加大 batch_size、確認 early stopping 真的在動')
    say('   （max_epochs=200、patience=20，若幾乎沒早停就是跑滿 200 輪）')

    # 一行摘要：無法把檔案帶出機器時照抄這一行即可
    say('')
    say('DIGEST N=%s T=%s L=%d H=%d | loop=%s batch=%s sp=%s | task=%s'
        % (N, T, args.lookback, args.hidden,
           ('%.2f' % _bench[0]) if _bench else '-',
           ('%.2f' % _bench[1]) if _bench else '-',
           ('%.1fx' % (_bench[0] / max(_bench[1], 1e-9))) if _bench else '-',
           args.task))


if __name__ == '__main__':
    main()