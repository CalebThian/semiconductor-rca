#!/usr/bin/env python
"""Residual-based anomaly detection via causal predictors.

Train MT-NGC task1 on Normal wafers, then measure per-sensor prediction
residuals on Abnormal wafers. High residual ratio = that sensor's causal
relationships broke down under abnormal conditions.

Handles case 1/3 (few Abnormal) because Abnormal only needs forward passes,
not graph fitting.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os

from src.datasets.semiconductor import load_semiconductor_csv, META_COLS
from src.models.mt_ngc_method import MTNeuralGrangerMethod
from scipy.stats import rankdata
from scripts.split_utils import block_split      # 依你的實際擺放位置調整

def seed_everything(seed):
    np.random.seed(seed)
    import random; random.seed(seed); torch.manual_seed(seed)


def split_by_symbol(csv_path):
    """回傳 (normal_data, abnormal_data, feature_names)。
    正規化統計量從 Normal 算,套用到兩者(陷阱:不可各自正規化)。"""
    for _enc in ('utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'gbk',
             'latin-1'):
        try:
            df = pd.read_csv(csv_path, encoding=_enc)
            if _enc not in ('utf-8-sig', 'utf-8'):
                print(f"  （{os.path.basename(csv_path)} 以 {_enc} 編碼讀取）")
            break
        except UnicodeDecodeError:
            continue
        else:
            raise ValueError(f"無法以常見編碼讀取 {csv_path}")
    sym = "Symbol Define"                       # 你的真欄名
    df[sym] = df[sym].astype(str)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    dt = "START_DATETIME"
    df[dt] = pd.to_datetime(df[dt], errors="coerce")
    df = df.dropna(subset=[dt]).sort_values(dt).reset_index(drop=True)

    normal_df = df[df[sym].str.lower().str.startswith("normal")]
    abn_df    = df[~df[sym].str.lower().str.startswith("normal")]
    print(f"  Normal {len(normal_df)}, Abnormal {len(abn_df)}")

    # 陷阱(a): Normal 的 μ/σ 套到兩者
   # ← 缺失值處理(與 loader 一致);3 個 NaN 就足以透過 MLP 傳染全部 sensor
    normal_feat = normal_df[feat_cols].ffill().bfill()
    abn_feat    = abn_df[feat_cols].ffill().bfill()
    n_nan_n = int(normal_feat.isna().sum().sum())
    n_nan_a = int(abn_feat.isna().sum().sum())
    if n_nan_n or n_nan_a:
        print(f"⚠ ffill/bfill 後仍有 NaN(整欄全空): Normal {n_nan_n}, Abnormal {n_nan_a} → 補 0")
        normal_feat = normal_feat.fillna(0.0)
        abn_feat    = abn_feat.fillna(0.0)

    # Normal 的 μ/σ 套到兩者(陷阱:不可各自正規化)
    mu = normal_feat.mean().values
    sd = normal_feat.std().values + 1e-8
    norm = ((normal_feat.values - mu) / sd).astype(np.float32)
    abn  = ((abn_feat.values    - mu) / sd).astype(np.float32)

    # 最後防線:確保沒有 NaN/inf 流進模型
    assert np.isfinite(norm).all(), "normal 資料仍有 NaN/inf!"
    assert np.isfinite(abn).all(),  "abnormal 資料仍有 NaN/inf!"
    return norm, abn, feat_cols


@torch.no_grad()
def task1_residuals(model, data, segment_lengths=None):
    """per-sensor 均方殘差 + per-sample 殘差矩陣。"""
    N = model.N
    per_sensor = np.zeros(N)
    per_sample_all = None
    for i in range(N):
        ds = model.make_dataset(data, "task1", i, segment_lengths=segment_lengths)
        n = len(ds)
        if n == 0:
            print(f"  ⚠ sensor {i}: dataset 為空!(資料太短,組不出 lookback 窗)")
        # --- 這裡是唯一未驗證的介面假設 ---
        X = torch.as_tensor(np.stack([ds[k][0] for k in range(len(ds))]), dtype=torch.float32)
        y = torch.as_tensor(np.array([ds[k][1] for k in range(len(ds))]), dtype=torch.float32).squeeze(-1)
        # ----------------------------------
        pred = model.predictors["task1"][i](X)
        resid = (y - pred) ** 2                      # (n_samples,)
        per_sensor[i] = resid.mean().item()
        if per_sample_all is None:
            per_sample_all = np.zeros((len(resid), N))
        per_sample_all[:, i] = resid.numpy()
    return per_sensor, per_sample_all


def graph_to_edges(g, names, top_frac=0.05, frac_base="all", thresh=None):
    """N x N 因果圖 -> 三欄 edge list: source,target,score,只保留分數最高的一小部分。

    方向約定:g[i, j] = j -> i(第 i 列的高分欄 j 是 i 的父節點),
    與下面 parents 的取法同源。若 get_graph 語意相反,在呼叫端傳 g.T。

    top_frac  : 保留比例(0.05 = 前 5%)。None 表示不裁切。
    frac_base : 'all'     -> 分母是所有可能的有向邊 N*(N-1)  (圖密度 5%)
                'nonzero' -> 分母是非零邊數                  (現有邊的前 5%)
    """
    N = len(names)
    arr = np.asarray(names, dtype=object)
    tgt_idx, src_idx = np.indices((N, N))
    df = pd.DataFrame({
        "source": arr[src_idx.ravel()],
        "target": arr[tgt_idx.ravel()],
        "score": np.asarray(g, dtype=float).ravel(),
    })
    df = df[(df.source != df.target) & np.isfinite(df.score) & (df.score != 0)]
    if thresh is not None:
        df = df[df.score >= thresh]
    df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)

    n_all, n_nonzero = N * (N - 1), len(df)
    if top_frac is not None:
        base = n_all if frac_base == "all" else n_nonzero
        n_keep = min(n_nonzero, max(1, int(np.ceil(top_frac * base))))
        print(f"  裁切:前 {top_frac:.1%} x {base} ({frac_base}) = {n_keep} 條,"
              f"非零邊共 {n_nonzero} / 可能有向邊 {n_all}")
        df = df.head(n_keep).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="single semiconductor CSV (has Symbol Define)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--lambda-sparse", type=float, default=0.1)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--n-blocks", type=int, default=10,
                    help="Normal 切成幾個連續區塊做 block split")
    ap.add_argument("--wafer-sensor-mult", type=float, default=3.0,
                    help="單片中某 sensor 殘差 > 基準的幾倍,算該 sensor 異常(預設 3x)")
    ap.add_argument("--w-nwafers", type=float, default=0.5,
                    help="n_anomalous_wafers 的權重(0~1);ratio 權重 = 1 - 此值。"
                         "0.5=各半, 0.7=以片數為主, 0.3=以 ratio 為主")
    ap.add_argument("--topk-parents", type=int, default=3,
                    help="每個 sensor 取幾個上游作為父節點(用於根因判定)")
    ap.add_argument("--root-thresh-mult", type=float, default=1.5,
                    help="ratio 超過中位數的幾倍才算異常(判定上游是否乾淨)")
    ap.add_argument("--onset-mult", type=float, default=2.0,
                    help="殘差超過基準幾倍算異常(時序偵測用)")
    ap.add_argument("--onset-consec", type=int, default=3,
                    help="需連續幾片超標才算真的開始異常(濾除單片雜訊)")
    ap.add_argument("--w-onset", type=float, default=0.7,
                    help="時序證據的權重(0=只看殘差, 1=只看時序)")
    ap.add_argument("--edge-top-frac", type=float, default=0.05,
                    help="edge list 只保留分數最高的前幾成(預設 0.05 = 前 5%%);0 或負數表示全留")
    ap.add_argument("--edge-frac-base", choices=["all", "nonzero"], default="all",
                    help="前 5%% 的分母:all=所有可能有向邊 N*(N-1)(圖密度), nonzero=現有非零邊")
    ap.add_argument("--edge-thresh", type=float, default=None,
                    help="裁切前先濾掉 score < 此值的邊")
    ap.add_argument("--edge-transpose", action="store_true",
                    help="若 get_graph 的語意是 g[i,j] = i -> j,加此旗標")
    args = ap.parse_args()

    normal, abnormal, names = split_by_symbol(args.csv)
    if len(abnormal) < 10:
        print(f"⚠ Abnormal only {len(abnormal)} wafers — per-wafer OK, per-sensor stats fragile")

    # 陷阱(1): 切 held-out Normal 當基準,不參與訓練
    # ⚠ 必須用「時間上連續的尾段」，不可隨機打散：
    #   task1 是 lookback 模型，打散後每個視窗的「歷史」都是隨機 wafer。
    seed_everything(args.seed)
    n_held = int(len(normal) * args.heldout_frac)
    n_train = len(normal) - n_held
    normal_train, train_seg, normal_held, held_seg = block_split(
        normal, heldout_frac=args.heldout_frac,
        n_blocks=args.n_blocks, lookback=args.lookback)
    print(f"  Normal train {len(normal_train)} (時間前段), "
          f"held-out {len(normal_held)} (時間後段)")


    # 1. Normal 上訓練 task1
    method = MTNeuralGrangerMethod(lookback=args.lookback, hidden_dim=64,
                                   lambda_sparse=args.lambda_sparse,
                                   max_epochs=args.max_epochs, patience=20, verbose=False)
    method.fit(normal_train, segment_lengths=train_seg)
    model = method.model                             # MultiTaskNeuralGranger

    g_normal = model.get_graph("task1")
    np.save(f"{args.out_prefix}_graph_normal_task1.npy", g_normal)

    # === CSV 0: 半導體 sensor 關係圖 (source,target,score),只留前 5% ===
    print(f"\n關係圖 edge list:")
    df_edges = graph_to_edges(g_normal.T if args.edge_transpose else g_normal, names,
                              top_frac=args.edge_top_frac if args.edge_top_frac > 0 else None,
                              frac_base=args.edge_frac_base, thresh=args.edge_thresh)
    edge_path = f"{args.out_prefix}_edges.csv"
    df_edges.to_csv(edge_path, index=False, encoding="utf-8")
    print(f"  -> {edge_path}(邊 {len(df_edges)} / 節點 {len(names)})")
    if len(df_edges):
        print(f"  score 範圍 {df_edges.score.min():.4g} ~ {df_edges.score.max():.4g}")
        print("  最強 3 條:" + "; ".join(
            f"{r.source}->{r.target} ({r.score:.4g})" for r in df_edges.head(3).itertuples()))

    # --- 漂移檢查：held-out 內部，時間前段 vs 後段 ---
    _, hd_mat = task1_residuals(model, normal_held, segment_lengths=held_seg)
    half = len(hd_mat) // 2
    early = float(np.median(hd_mat[:half]))
    late  = float(np.median(hd_mat[half:]))
    print(f"  [漂移檢查] held-out 前段 median {early:.6g} / "
          f"後段 {late:.6g} / 比值 {late/(early+1e-12):.3f}")

    # 2. 殘差:held-out Normal(基準) vs Abnormal
    r_base, _        = task1_residuals(model, normal_held)
    r_abn, abn_perwf = task1_residuals(model, abnormal)

    # 3. 陷阱(2): 比值,不是絕對殘差
    ratio = r_abn / (r_base + 1e-8)

    # 先算 per-wafer × per-sensor 異常判定(若 wafer 段已算過就複用,別重算)
    anomaly_thresh = r_base * args.wafer_sensor_mult          # 每 sensor 的異常門檻
    is_anom = abn_perwf > anomaly_thresh[None, :]             # (n_abnormal_wafers, N) bool

    # ========== 時序證據:誰先開始異常 ==========
    # 根因在時間上領先其下游 —— 這是與殘差幅度獨立的證據
    def detect_onset(resid_mat, base, mult, consec):
        """回傳每個 sensor 首次「持續」超標的 wafer index;未超標為 inf。"""
        T, N = resid_mat.shape
        thr = base * mult
        onset = np.full(N, np.inf)
        for i in range(N):
            if not np.isfinite(thr[i]) or thr[i] <= 0:
                continue
            run = 0
            for t in range(T):
                run = run + 1 if resid_mat[t, i] > thr[i] else 0
                if run >= consec:
                    onset[i] = t - consec + 1
                    break
        return onset

    onset = detect_onset(abn_perwf, r_base,
                         args.onset_mult, args.onset_consec)
    n_detected = int(np.isfinite(onset).sum())
    print(f"\n時序偵測:{n_detected}/{len(names)} 個 sensor 出現持續異常"
          f"(門檻 {args.onset_mult}x 基準,連續 {args.onset_consec} 片)")
    if n_detected:
        earliest = int(np.nanmin(onset[np.isfinite(onset)]))
        first = [names[i] for i in np.where(onset == earliest)[0]]
        print(f"  最早異常於第 {earliest} 片:{', '.join(first)}")

    # 時序分數:越早異常越可疑
    from scipy.stats import rankdata
    N_s = len(names)
    onset_filled = np.where(np.isfinite(onset), onset, abn_perwf.shape[0] + 1)
    s_onset = 1.0 - rankdata(onset_filled) / N_s        # 越早 → 越高
    s_ratio = rankdata(np.where(np.isfinite(ratio), ratio, 0)) / N_s

    w = args.w_onset
    temporal_score = w * s_onset + (1.0 - w) * s_ratio

    # ========== 根因判定:用因果圖區分「源頭」與「下游污染」 ==========
    # 原理:殘差會沿因果結構傳染 —— 根因的值跑掉,下游收到異常輸入,
    #      預測也跟著失準。故純殘差排名會給出一整團異常 sensor。
    #   殘差大 + 父節點正常 -> 擾動由此進入系統(根因)
    #   殘差大 + 父節點也大 -> 只是下游污染
    g_root = model.get_graph("task1")            # 與殘差同源的 Normal 圖
    N = len(names)
    gg = g_root.copy()
    np.fill_diagonal(gg, -np.inf)

    K = args.topk_parents
    parents = {i: [int(j) for j in np.argsort(gg[i])[::-1][:K]
                   if np.isfinite(gg[i, j])] for i in range(N)}

    # 父節點修正分數:自身異常程度 ÷ 上游最嚴重的異常程度
    parent_max = np.array([
        max([ratio[p] for p in parents[i] if np.isfinite(ratio[p])], default=1.0)
        for i in range(N)])
    root_score = ratio / (parent_max + 1e-8)

    # 上游是否乾淨(自身超標但所有父節點都未超標)
    thr = np.nanmedian(ratio) * args.root_thresh_mult
    is_root_candidate = np.array([
        bool(np.isfinite(ratio[i]) and ratio[i] > thr
             and all(ratio[p] < thr for p in parents[i]))
        for i in range(N)])

    print(f"\n根因判定:門檻 ratio > {thr:.3f}(中位數 x {args.root_thresh_mult})")
    print(f"  超標 sensor: {int((ratio > thr).sum())} 個")
    print(f"  其中上游乾淨(根因候選): {int(is_root_candidate.sum())} 個")


   # === CSV 1: sensor 異常排名 + 異常次數 ===
    n_anom_wafers = is_anom.sum(axis=0)                       # ← 每個 sensor 在幾片 wafer 上異常
    n_abn_total = abn_perwf.shape[0]                          # Abnormal wafer 總數

    med = np.median(r_base[~np.isnan(r_base)])
    mad = np.median(np.abs(r_base[~np.isnan(r_base)] - med)) + 1e-8
    rbase_outlier = (r_base - med) / (1.4826 * mad) > 3      # robust z > 3

    # 假 0 = ratio 高 + n=0 + r_base 是離群值
    fake_zero = (n_anom_wafers == 0) & (ratio > 3) & rbase_outlier
    valid = ~np.isnan(ratio)
    n_ok = (valid & ~fake_zero).sum()
    if n_ok == 0:
        print("⚠ 過濾後沒有任何可用 sensor — 過濾器可能過於激進,改為輸出全部並標記")
        fake_zero[:] = False
    
    df_s = pd.DataFrame({
        "sensor": names,
        "resid_abnormal": r_abn,
        "resid_normal_base": r_base,
        "ratio": ratio,
        "n_anomalous_wafers": n_anom_wafers,                  # ← 異常次數(片數)
        "pct_anomalous": n_anom_wafers / n_abn_total,         # ← 異常比例(佔 Abnormal 的幾成)
        "r_base": r_base,
        "flag": np.where(fake_zero, "unreliable_threshold",
                np.where(np.isnan(ratio), "dead/nan", "ok")),
        "parent_max_ratio": parent_max,
        "root_score": root_score,
        "is_root_candidate": is_root_candidate,
        "top_parents": [", ".join(names[p] for p in parents[i]) for i in range(N)],
        "onset_wafer": onset,               # 首次持續異常的 wafer index
        "temporal_score": temporal_score,   # 時序+殘差 組合分數
    })
    n_fake = fake_zero.sum()
    if n_fake:
        print(f"⚠ {n_fake} 個 sensor 門檻不可信(r_base 離群),標記為 unreliable 並排除 ranking")
    
    # === 加權排名:n_anomalous_wafers + ratio ===
    # 各自轉成 percentile rank(0~1,越高越異常),對極端值穩健
    # 只對 flag=='ok' 的 sensor 排名(排除假 0 / dead)
    ok = (df_s.flag == "ok").values
    score = np.full(len(df_s), np.nan)

    r_ratio = rankdata(df_s.loc[ok, "ratio"], method="average") / ok.sum()
    r_nwaf  = rankdata(df_s.loc[ok, "n_anomalous_wafers"], method="average") / ok.sum()

    w_nwaf, w_ratio = args.w_nwafers, 1.0 - args.w_nwafers
    score[ok] = w_nwaf * r_nwaf + w_ratio * r_ratio

    df_s["rank_score"] = score
    df_s["pctile_ratio"] = np.nan;    df_s.loc[ok, "pctile_ratio"] = r_ratio
    df_s["pctile_nwafers"] = np.nan;  df_s.loc[ok, "pctile_nwafers"] = r_nwaf

    # 排名:只排 ok 的,依 rank_score 降序
    df_rank = df_s[ok].sort_values("rank_score", ascending=False).reset_index(drop=True)
    df_rank.insert(0, "rank", df_rank.index + 1)
    df_rank.to_csv(f"{args.out_prefix}_anomaly_sensors.csv", index=False)
    df_s.to_csv(f"{args.out_prefix}_anomaly_sensors_full.csv", index=False)   # 含被排除的

    # 根因排名(待 SVAR 驗證,暫不取代 ratio 排名)
    ok_mask = (df_s.flag == "ok") if "flag" in df_s.columns else np.ones(len(df_s), bool)
    df_root = (df_s[ok_mask]
               .sort_values("root_score", ascending=False)
               .reset_index(drop=True))
    df_root.insert(0, "root_rank", df_root.index + 1)
    df_root.to_csv(f"{args.out_prefix}_root_cause.csv", index=False)

    df_temporal = (df_s[ok_mask]
                   .sort_values("temporal_score", ascending=False)
                   .reset_index(drop=True))
    df_temporal.insert(0, "temporal_rank", df_temporal.index + 1)
    df_temporal.to_csv(f"{args.out_prefix}_root_cause_temporal.csv", index=False)

    # 兩種排名的差異 —— 差異大代表殘差傳染顯著,圖修正有實質作用
    top5_ratio = set(df_s[ok_mask].nlargest(5, "ratio").sensor)
    top5_root = set(df_root.head(5).sensor)
    overlap = len(top5_ratio & top5_root)
    print(f"\n=== 兩種排名的 top-5 重疊: {overlap}/5 ===")
    if overlap <= 2:
        print("  差異大 -> 殘差傳染顯著,圖修正改變了結論(值得深究)")
    elif overlap >= 4:
        print("  差異小 -> 異常分散,無明顯傳播鏈(或圖未提供額外資訊)")
    print(f"  ratio 排名 top-5     : {', '.join(df_s[ok_mask].nlargest(5,'ratio').sensor)}")
    print(f"  root_score 排名 top-5: {', '.join(df_root.head(5).sensor)}")

    # === CSV 2: wafer 異常排名 + 異常參數計數 ===
    # 每個 sensor 的 Normal 基準(per-sensor,用來判定單片是否異常)
    # r_base 已是 per-sensor 均值;用它當每個 sensor 的正常水位
    anomaly_thresh = r_base * args.wafer_sensor_mult          # 超過基準 N 倍算該 sensor 異常
    # abn_perwf shape = (n_abnormal_wafers, N):每片每 sensor 的殘差
    is_anom = abn_perwf > anomaly_thresh[None, :]            # (wafers, N) bool
    n_anom_sensors = is_anom.sum(axis=1)                     # 每片有幾個 sensor 異常
    wf_score = abn_perwf.sum(axis=1)                         # 每片總殘差

    df_w = pd.DataFrame({
        "abnormal_wafer_idx": np.arange(len(wf_score)),
        "total_residual": wf_score,
        "n_anomalous_sensors": n_anom_sensors,               # ← 新增:異常參數總數
        "top_sensor": [names[k] for k in abn_perwf.argmax(axis=1)],
        "top_sensor_residual": abn_perwf.max(axis=1),        # ← 新增:最爆那個的殘差值
    }).sort_values("total_residual", ascending=False).reset_index(drop=True)
    df_w.insert(0, "rank", df_w.index + 1)
    df_w.to_csv(f"{args.out_prefix}_anomaly_wafers.csv", index=False)

    

if __name__ == "__main__":
    main()