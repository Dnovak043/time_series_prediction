# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:50:15 2026

@author: vanio
"""

from __future__ import annotations
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import array


import pandas as pd
import datetime
import pickle
from read_databento_new import dbn_to_df
#------------------------------------------------------------------------------
# Retrive data frame with real-valued time series with ALL features
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def add_event_features_and_resample_time(
    df: pd.DataFrame,
    dt_sec: int = 1,                 # bucket size in seconds
    W_events: int = 300,             # trailing event-vol window on raw events (optional)
    k_fwdL = [1,2,3],        # forward horizons in *buckets*
    sort_cols=("ts_event", "sequence"),
    fill_empty_buckets: bool = True, # create a regular clock grid and ffill snapshot
) -> pd.DataFrame:
    """
    Clock-time bars from event stream:
      - Each dt_sec bucket is represented by the last event in the bucket (snapshot columns).
      - Block features are computed using the set of events inside that bucket.
      - Optional: fill empty buckets by forward-filling snapshot and setting event-based counts to 0.
Features:
'mid_price'
'log_mid' 
'log_mid_ret' 
'sigma_W' 
'mid_changed' 
'mid_change_count_n'   -number of midprice chages in the block - 0 ..n
'move_rate_per_event'   normalized number of chages per event
'ofi_L1'                order flow imbalance
'ofi_L1_n'              rolling OFI over last n events
'dt_block_sec'          seconds per block
'log_dt_block'
'event_intensity'       events per second
'log_event_intensity'
'move_rate_per_sec'     mid price movements per second
'trade_buy_vol_event'   the trade size at time t if the trade was buy-initiated
'trade_sell_vol_event'  the trade size at time t if the trade was sell-initiated  
'trade_buy_vol_n'       last n events
'trade_sell_vol_n'      last n events
'trade_ofi_n'           rolling signed trade volume imbalance over last n events
'vpin'                  volume-bucket VPIN on trades
'event_idx'
'sample_idx'
'log_mid_return_fwd_1'
'log_mid_sum_bwd_1'
'log_mid_sum_fwd_1'
'log_mid_sum_ratio_1'
'imbalance'
'micro_price'
'spread'
'rel_spread'
'log_spread'
'micro_mid'
'micro_mid_sign'
'mid_cross_prev_ask_up'
'mid_cross_prev_bid_dn'
'mid_ret'
'jump_gt_prev_spread'
    """

    df2 = df.sort_values(list(sort_cols)).copy()

    tcol = "ts_event"

    # --- forward-fill LOB snapshot columns at event level ---
    lob_prefixes = ("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_")
    lob_cols = [c for c in df2.columns if c.startswith(lob_prefixes)]
    df2[lob_cols] = df2[lob_cols].ffill()

    # --- event-level mid/log/returns ---
    df2["mid_price"] = (df2["bid_px_00"].astype("float64") + df2["ask_px_00"].astype("float64")) / 2.0
    df2["log_mid"] = np.log(df2["mid_price"].where(df2["mid_price"] > 0))
    # NOTE: this diff crosses bucket boundaries; for within-bucket vol we compute separately below
    df2["log_mid_ret"] = df2["log_mid"].diff()

    # --- optional trailing event-vol sigma_W on raw events (RMS over last W_events returns) ---
    if W_events is not None and W_events > 0:
        df2["sigma_W"] = (
            df2["log_mid_ret"]
               .rolling(W_events, min_periods=W_events)
               .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True)
        )
    else:
        df2["sigma_W"] = np.nan

    # --- assign each event to a time bucket (bucket_start) ---
    freq = f"{int(dt_sec)}S"
    df2["bucket_start"] = df2[tcol].dt.floor(freq)

    # --- group by bucket and compute:
    #     - last snapshot row (rep bar)
    #     - within-bucket counts/vol/intensity
    g = df2.groupby("bucket_start", sort=True)

    # (A) representative snapshot = last event in bucket
    last_cols = [tcol] + lob_cols + ["mid_price", "log_mid", "sigma_W"]
    dfN = g[last_cols].last().copy()

    # store last-event timestamp explicitly
    dfN.rename(columns={tcol: "ts_last_event"}, inplace=True)

    # (B) within-bucket event_count
    dfN["event_count"] = g.size().astype(np.int32)

    # (C) within-bucket duration (span between first and last event in the bucket)
    #     (can be 0 if only 1 event; for empty buckets we'll set 0)
    t_first = g[tcol].first()
    t_last = g[tcol].last()
    dfN["dt_span_sec"] = (t_last - t_first).dt.total_seconds().astype("float64")

    # (D) within-bucket mid-price change count (changes *inside* bucket)
    #     use diffs inside each bucket to avoid boundary artifacts
    def _mid_change_count_in_bucket(s: pd.Series) -> int:
        v = s.to_numpy()
        if v.size <= 1:
            return 0
        return int(np.sum(v[1:] != v[:-1]))

    dfN["mid_change_count"] = g["mid_price"].apply(_mid_change_count_in_bucket).astype(np.int16)

    # (E) within-bucket realized vol proxy on log_mid, computed *inside* bucket
    #     returns = diff within bucket, RMS = sqrt(mean(r^2))
    def _sigma_bucket_from_logmid(s: pd.Series) -> float:
        v = s.to_numpy(dtype=float)
        if v.size <= 1:
            return np.nan
        r = np.diff(v)
        return float(np.sqrt(np.mean(r * r)))

    dfN["sigma_bucket"] = g["log_mid"].apply(_sigma_bucket_from_logmid).astype("float64")

    # (F) rates / intensity (use fixed dt_sec for a stable clock-time definition)
    eps = 1e-12
    dfN["dt_bucket_sec"] = float(dt_sec)
    dfN["event_intensity"] = dfN["event_count"] / dfN["dt_bucket_sec"]
    dfN["log_event_intensity"] = np.log(dfN["event_intensity"].clip(lower=eps))
    dfN["move_rate_per_event"] = dfN["mid_change_count"] / dfN["event_count"].replace(0, np.nan)
    dfN["move_rate_per_sec"] = dfN["mid_change_count"] / dfN["dt_bucket_sec"]

    # --- optionally fill empty buckets to get a regular clock grid ---
    if fill_empty_buckets:
        full_idx = pd.date_range(dfN.index.min(), dfN.index.max(), freq=freq, tz=df2[tcol].dt.tz)
        dfN = dfN.reindex(full_idx)

        # if no events in a bucket: event-based quantities become 0, snapshot forward-fills
        dfN["event_count"] = dfN["event_count"].fillna(0).astype(np.int32)
        dfN["mid_change_count"] = dfN["mid_change_count"].fillna(0).astype(np.int16)
        dfN["dt_span_sec"] = dfN["dt_span_sec"].fillna(0.0)

        # forward fill snapshot/state features from previous bucket
        dfN[lob_cols + ["mid_price", "log_mid", "sigma_W"]] = dfN[lob_cols + ["mid_price", "log_mid", "sigma_W"]].ffill()
        dfN["ts_last_event"] = dfN["ts_last_event"].ffill()

        # recompute derived rates safely after filling
        dfN["event_intensity"] = dfN["event_count"] / float(dt_sec)
        dfN["log_event_intensity"] = np.log(dfN["event_intensity"].clip(lower=eps))
        dfN["move_rate_per_event"] = dfN["mid_change_count"] / dfN["event_count"].replace(0, np.nan)
        dfN["move_rate_per_sec"] = dfN["mid_change_count"] / float(dt_sec)

    # index is bucket_start (clock grid)
    dfN.index.name = "bucket_start"
    dfN["sample_idx"] = np.arange(len(dfN), dtype=np.int64)

    # --- forward targets on the clock-time series (k_fwd in buckets) ---
    for k_fwd in k_fwdL:
        k = int(k_fwd)
        s = dfN["log_mid"]

        dfN[f"log_mid_return_fwd_{k}"] = s.shift(-k) - s
        dfN[f"log_mid_sum_bwd_{k}"] = s.rolling(window=k, min_periods=k).sum()
        dfN[f"log_mid_sum_fwd_{k}"] = s.shift(-1).rolling(window=k, min_periods=k).sum().shift(-(k-1))

        bwd = dfN[f"log_mid_sum_bwd_{k}"]
        dfN[f"log_mid_sum_ratio_{k}"] = (dfN[f"log_mid_sum_fwd_{k}"] - bwd) / bwd.replace(0, np.nan)

    # --- LOB derived features on the clock-time snapshots ---
    den = (dfN["bid_sz_00"] + dfN["ask_sz_00"]).replace(0, np.nan)
    dfN["imbalance"] = dfN["bid_sz_00"] / den
    dfN["micro_price"] = dfN["imbalance"] * dfN["ask_px_00"] + (1.0 - dfN["imbalance"]) * dfN["bid_px_00"]

    dfN["spread"] = dfN["ask_px_00"] - dfN["bid_px_00"]
    dfN["rel_spread"] = dfN["spread"] / dfN["mid_price"].replace(0, np.nan)

    eps_p = 1e-12
    dfN["log_spread"] = np.log(dfN["ask_px_00"].clip(lower=eps_p)) - np.log(dfN["bid_px_00"].clip(lower=eps_p))

    dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
    dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])

    prev_bid = dfN["bid_px_00"].shift(1)
    prev_ask = dfN["ask_px_00"].shift(1)
    dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
    dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)

    dfN["mid_ret"] = dfN["mid_price"].diff()
    prev_spread = dfN["spread"].shift(1)
    dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)

    return dfN

#-----------------------------------------------------------------------------


#------------------------------------------------------------------------------
def compute_ofi_level_k(
    df: pd.DataFrame,
    k: int = 0,
    n: int | None = None,
    prefix: str = "ofi",
    max_levels: int = 10,
) -> pd.Series:
    """
    Compute event-level OFI at book level k without modifying df.

    If n is provided, return the rolling n-event OFI sum.
    """
    if not 0 <= k < max_levels:
        raise ValueError(
            f"k must be in [0, {max_levels - 1}] for MBP-{max_levels}."
        )

    bp = df[f"bid_px_{k:02d}"].to_numpy(dtype=float)
    ap = df[f"ask_px_{k:02d}"].to_numpy(dtype=float)
    bq = df[f"bid_sz_{k:02d}"].to_numpy(dtype=float)
    aq = df[f"ask_sz_{k:02d}"].to_numpy(dtype=float)

    bp_prev = np.roll(bp, 1)
    ap_prev = np.roll(ap, 1)
    bq_prev = np.roll(bq, 1)
    aq_prev = np.roll(aq, 1)

    # The first observation has no previous book state.
    bp_prev[0] = np.nan
    ap_prev[0] = np.nan
    bq_prev[0] = np.nan
    aq_prev[0] = np.nan

    db = np.where(
        bp > bp_prev,
        bq,
        np.where(
            bp < bp_prev,
            -bq_prev,
            bq - bq_prev,
        ),
    )

    da = np.where(
        ap < ap_prev,
        aq,
        np.where(
            ap > ap_prev,
            -aq_prev,
            aq - aq_prev,
        ),
    )

    ofi_k = pd.Series(
        db - da,
        index=df.index,
        name=f"{prefix}_{k}",
        dtype=float,
    ).fillna(0.0)

    if n is not None and n > 0:
        ofi_k = ofi_k.rolling(
            n,
            min_periods=1,
        ).sum()

        ofi_k.name = f"{prefix}_{k}_n"

    return ofi_k

def add_ofi_multi_level(
    df2: pd.DataFrame,
    L: int = 10,
    n: int | None = None,
    weights: str = "exp",
    tau: float = 3.0,
    prefix: str = "ofi",
) -> pd.DataFrame:
    """
    Return df2 with:

      - f"{prefix}_L{L}"   : weighted sum of OFI_k, k = 0, ..., L-1
      - f"{prefix}_L{L}_n" : rolling sum over the last n events, if requested

    The individual OFI_k series are computed temporarily and are not added
    to df2.
    """
    if L <= 0:
        raise ValueError("L must be positive.")

    L = min(int(L), 10)  # MBP-10

    # Construct level weights.
    levels = np.arange(L, dtype=float)

    if weights == "uniform":
        w = np.ones(L, dtype=float)
    elif weights == "inv":
        w = 1.0 / (levels + 1.0)
    elif weights == "exp":
        if tau <= 0:
            raise ValueError("tau must be greater than zero for exp weights.")
        w = np.exp(-levels / float(tau))
    else:
        raise ValueError(
            "weights must be one of: 'uniform', 'inv', or 'exp'."
        )

    # Preserve a comparable scale for different values of L.
    w /= w.sum()

    # Compute the weighted multi-level OFI without inserting temporary columns.
    ofi_sum = np.zeros(len(df2), dtype=float)

    for k, weight in enumerate(w):
        ofi_k =  compute_ofi_level_k(
            df2,
            k=k,
            n=None,
            prefix=prefix,
            max_levels=10,
        )

        ofi_sum += weight * ofi_k.to_numpy(
            dtype=float,
            copy=False,
        )

    name = f"{prefix}_L{L}"

    new_columns = {
        name: pd.Series(
            ofi_sum,
            index=df2.index,
            dtype=float,
        )
    }

    if n is not None and n > 0:
        new_columns[f"{name}_n"] = (
            new_columns[name]
            .rolling(n, min_periods=n)
            .sum()
        )

    # Remove previous versions, if present, and append all new columns once.
    output_columns = list(new_columns)

    result = df2.drop(
        columns=output_columns,
        errors="ignore",
    )

    result = pd.concat(
        [
            result,
            pd.DataFrame(new_columns, index=df2.index),
        ],
        axis=1,
    )

    return result


#------------------------------------------------------------------------------
def make_ofi_weights(L: int = 10, weights: str = "exp", tau: float = 3.0):
    L = min(int(L), 10)

    if weights == "uniform":
        w = np.ones(L, dtype=float)

    elif weights == "inv":
        w = 1.0 / (np.arange(L, dtype=float) + 1.0)

    elif weights == "exp":
        if tau <= 0:
            raise ValueError("tau must be > 0 for exp weights.")
        w = np.exp(-np.arange(L, dtype=float) / float(tau))

    else:
        raise ValueError("weights must be 'uniform', 'inv', or 'exp'.")

    return w / w.sum()
#------------------------------------------------------------------------------
def add_normalized_ofi_multi_level(
    df2,
    L: int = 10,
    n: int | None = 10,
    weights: str = "exp",
    tau: float = 3.0,
    prefix: str = "ofi",
    eps: float = 1e-12,
):
    """
    Adds:
      - ofi_L10_norm   : weighted deep OFI divided by weighted depth
      - ofi_L10_norm_n : rolling weighted deep OFI divided by rolling weighted depth
    """

    L = min(int(L), 10)

    # First compute weighted deep OFI event-level:
    # df2[f"{prefix}_L{L}"] = sum_k w_k OFI_k
    df2 = add_ofi_multi_level(
        df2,
        L=L,
        n=None,
        weights=weights,
        tau=tau,
        prefix=prefix,
    )

    name = f"{prefix}_L{L}"
    ofi = df2[name].astype(float)

    # Same weights as numerator
    w = make_ofi_weights(L=L, weights=weights, tau=tau)

    bid_cols = [f"bid_sz_{k:02d}" for k in range(L)]
    ask_cols = [f"ask_sz_{k:02d}" for k in range(L)]

    weighted_bid_depth = df2[bid_cols].astype(float).mul(w, axis=1).sum(axis=1)
    weighted_ask_depth = df2[ask_cols].astype(float).mul(w, axis=1).sum(axis=1)

    depth = (weighted_bid_depth + weighted_ask_depth).clip(lower=eps)

    # event-level normalized weighted deep OFI
    df2[f"{name}_norm"] = ofi / depth

    if n is not None and n > 0:
        # rolling normalized weighted deep OFI
        # normalize after summing numerator and denominator over the same window
        ofi_n = ofi.rolling(n, min_periods=1).sum()
        depth_n = depth.rolling(n, min_periods=1).sum()

        df2[f"{name}_norm_n"] = ofi_n / depth_n.clip(lower=eps)

    return df2

#------------------------------------------------------------------------------
def add_event_features_and_resample(
    df: pd.DataFrame,
    n: int = 10,                          # events resampling window
    W: int = 300,                         # back window for volatility estimATION
    k_fwdL = [1,2,3],                     # forward  windows for the predicted feature
    sort_cols=("ts_event", "sequence"),
    take: str = "last",                   # "last" keeps 10th/20th/... ; "first" keeps 1st/11th/...
    # VPIN params
    compute_vpin: bool = True,
    vpin_bucket_vol: float = 10000, #50_000.0,    # volume bucket size (shares/contracts)
    vpin_mavg: int = 10 # 20, 50,                  # rolling avg over last m buckets
) -> pd.DataFrame:

    if take not in {"last", "first"}:
        raise ValueError("take must be 'last' or 'first'")

    df2 = df.sort_values(list(sort_cols)).copy()
    tcol = "ts_event"  # tz-aware UTC datetime per Databento


    # --- forward-fill LOB snapshot columns (so every row has a complete state) ---
    lob_prefixes = ("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_")
    lob_cols = [c for c in df2.columns if c.startswith(lob_prefixes)]
    df2[lob_cols] = df2[lob_cols].ffill()

    # --- mid / log-mid / event returns (raw event series) ---
    df2["mid_price"] = (df2["bid_px_00"].astype("float64") + df2["ask_px_00"].astype("float64")) / 2.0
    df2["log_mid"] = np.log(df2["mid_price"].where(df2["mid_price"] > 0))
    df2["log_mid_ret"] = df2["log_mid"].diff()

    # --- event-volatility sigma_W at every raw event (RMS over last W event-returns) ---
    df2["sigma_W"] = (
        df2["log_mid_ret"]
           .rolling(W, min_periods=W)
           .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True)
    )
    

    # --- block features over last n raw events (aligned to each raw event) ---
    # Mid-price changes count over last n raw events
    dmid = df2["mid_price"].diff()
    df2["mid_changed"] = dmid.fillna(0).ne(0).astype(np.int8)
    
    df2["mid_change_count_n"] = (
        df2["mid_changed"]
           .rolling(n, min_periods=1)
           .sum()
           .astype(np.int16)
    )
    
    df2["move_rate_per_event"] = df2["mid_change_count_n"] / n
#------------------------------------------------------------------------------
    # --- L1 OFI (Cont-style) per event, then rolling sum over last n events ---
    bpx = df2["bid_px_00"].astype(float)
    apx = df2["ask_px_00"].astype(float)
    bsz = df2["bid_sz_00"].astype(float)
    asz = df2["ask_sz_00"].astype(float)

    bpx_prev, apx_prev = bpx.shift(1), apx.shift(1)
    bsz_prev, asz_prev = bsz.shift(1), asz.shift(1)

    db = np.where(bpx > bpx_prev,  bsz,
         np.where(bpx < bpx_prev, -bsz_prev,
                  bsz - bsz_prev))
    da = np.where(apx < apx_prev,  asz,
         np.where(apx > apx_prev, -asz_prev,
                  asz - asz_prev))
    df2["ofi_L1"] = (db - da)
    df2["ofi_L1"] = df2["ofi_L1"].fillna(0)
    # order flow imbalance: Flow (LOB): ofi_L1_n (rolling OFI over last n events)
#   df2["ofi_L1_n"] = pd.Series(df2["ofi_L1"], index=df2.index).rolling(n, min_periods=n).sum()
#    df2["ofi_L1_n"] = df2["ofi_L1"].rolling(window=n, min_periods=1).sum()
    


    # 1. Compute your clean rolling sum exactly as you have it
    df2["ofi_L1_n"] = df2["ofi_L1"].rolling(window=n, min_periods=1).sum()
    
    # 2. Compute the simultaneous total depth at Level 1 per event
    total_depth_event = df2["bid_sz_00"] + df2["ask_sz_00"]
    
    # 3. Get the average depth across that exact same backward window
    # Note: Match the window and min_periods parameters perfectly
    rolling_avg_depth = total_depth_event.rolling(window=n, min_periods=1).mean()
    
    # 4. Safe division to prevent dividing by zero if depth vanishes
    df2["ofi_L1_n_norm"] = df2["ofi_L1_n"] / rolling_avg_depth.replace(0, np.nan)
    df2["ofi_L1_n_norm"] = df2["ofi_L1_n_norm"].fillna(0.0)



    
    
#------------------------------------------------------------------------------
    # Physical duration of last n events: τ_t - τ_{t-n+1} 
    # Using diff(periods=n-1) gives τ_t - τ_{t-(n-1)}
    # duration/intensity over last n events
    eps_time = 1e-9
    df2["dt_block_sec"] = df2[tcol].diff(n - 1).dt.total_seconds()
    df2["dt_block_sec"] = df2["dt_block_sec"].clip(lower=eps_time)

    df2["log_dt_block"] = np.log(df2["dt_block_sec"])
    df2["event_intensity"] = n / df2["dt_block_sec"]
    df2["log_event_intensity"] = np.log(df2["event_intensity"])
    df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"]

#------------------------------------------------------------------------------
    # --- Trade-based rolling OFI over last n events (uses action/side/size) ---
    a = df2["action"].astype(str).str.upper()
    side = df2["side"].astype(str).str.upper()
    is_trade = a.str.startswith("T")  # adjust if your trade action differs
    size = df2["size"].astype(float).to_numpy()
    price = df2["price"].astype(float).to_numpy()
    mid = df2["mid_price"].to_numpy()
    
    # aggressor sign: +1 buy, -1 sell
    sgn = np.zeros(len(df2), dtype=np.int8)
    sgn[is_trade.values & (side.values == "B")] = 1
    sgn[is_trade.values & (side.values == "A")] = -1

    # fallback for side == 'N' on trade rows: midpoint/tick rule
    unk = is_trade.values & (side.values == "N")
    if np.any(unk):
        s = np.sign(price[unk] - mid[unk]).astype(np.int8)
        tick = np.sign(np.diff(price, prepend=price[0]))[unk].astype(np.int8)
        s = np.where(s != 0, s, tick)
        # carry last nonzero within the *trade stream* (simple pass)
        last = 0
        for i, val in enumerate(s):
            if val == 0:
                s[i] = last
            else:
                last = val
        sgn[unk] = s

    buy_vol_event = np.where(is_trade.values & (sgn > 0), size, 0.0)
    sell_vol_event = np.where(is_trade.values & (sgn < 0), size, 0.0)

    df2["trade_buy_vol_event"] = buy_vol_event
    df2["trade_sell_vol_event"] = sell_vol_event

    df2["trade_buy_vol_n"] = pd.Series(buy_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    df2["trade_sell_vol_n"] = pd.Series(sell_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    tv = df2["trade_buy_vol_n"] + df2["trade_sell_vol_n"]
    # Flow (trades): trade_ofi_n (rolling signed trade volume imbalance over last n events)
    df2["trade_ofi_n"] = ((df2["trade_buy_vol_n"] - df2["trade_sell_vol_n"]) / tv.replace(0, np.nan)).fillna(0)
    df2["tvi_n"] =       ((df2["trade_buy_vol_n"] - df2["trade_sell_vol_n"]) / tv.replace(0, np.nan)).fillna(0)
    # --- VPIN (volume-bucket VPIN on trades), forward-filled to events ---
    df2["vpin"] = np.nan
    if compute_vpin:
        tr_idx = np.flatnonzero(is_trade.values)
        if tr_idx.size > 0:
            tr_size = size[tr_idx]
            tr_buy = buy_vol_event[tr_idx]
            tr_sell = sell_vol_event[tr_idx]

            cumv = np.cumsum(tr_size)
            bucket_id = (cumv // float(vpin_bucket_vol)).astype(np.int64)

            buy_b = pd.Series(tr_buy).groupby(bucket_id).sum()
            sell_b = pd.Series(tr_sell).groupby(bucket_id).sum()

            imb = (buy_b - sell_b).abs()
            vpin_bucket = (imb / float(vpin_bucket_vol)).rolling(vpin_mavg, min_periods=vpin_mavg).mean()

            # map bucket VPIN back to each trade, then ffill to all events
            vpin_trade = pd.Series(bucket_id).map(vpin_bucket).to_numpy()
            #df2.loc[df2.index[tr_idx], "vpin"] = vpin_trade
            col = df2.columns.get_loc("vpin")
            df2.iloc[tr_idx, col] = vpin_trade
            df2["vpin"] = df2["vpin"].ffill()
            df2["vpin"] = df2["vpin"].bfill()
            df2["dvpin"]= df2["vpin"].diff()
            df2["dvpin"]= df2["dvpin"].bfill()
#------------------------------------------------------------------------------
    # Optional: mid-change rate per second
    df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"].clip(lower=eps_time)


    ofi_window = n         # For scale-invariance / fractal-style analysis
    # ofi_window = 10      # For predictive modeling  

    df2 = add_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")
    #deep_ofi = df2["ofi_L10_n"]


    # noprmalized ofi
    df2 = add_normalized_ofi_multi_level(df2, L=1, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")

    df2 = add_normalized_ofi_multi_level(df2, L=3, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")

    df2 = add_normalized_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")

    

    # --- downsample every n events (keep 'first' or 'last' in each n-block) ---
    ev = np.arange(len(df2), dtype=np.int64)
    if take == "first":
        mask = (ev % n) == 0
    else:
        mask = (ev % n) == (n - 1)
    dfN = df2.loc[mask].copy()
    
    dfN = dfN.sort_index()  # timestamp index
    
    dfN["event_idx"] = ev[mask]
    dfN["sample_idx"] = (dfN["event_idx"] // n).astype(np.int64)
    # -------------------------------------------------------------------------
    
    
    # k-step-ahead log return on the resampled series 
    # --- forward k-step return on the RESAMPLED series (optional) ---
    for k_fwd in k_fwdL:
        s = dfN["log_mid"]
    
        # forward k-step return: log_mid[t+k] - log_mid[t]
        dfN[f"log_mid_return_fwd_{k_fwd}"] = s.shift(-k_fwd) - s
        dfN[f"log_mid_return_fwd_{k_fwd}"].replace( np.nan,0)
        # backward sum: log_mid[t-k+1] + ... + log_mid[t]
        dfN[f"log_mid_sum_bwd_{k_fwd}"] = s.rolling(window=k_fwd, min_periods=k_fwd).sum()
    
        # forward sum: log_mid[t+1] + ... + log_mid[t+k]
        # rolling is backward-looking => shift result by -(k-1)
        dfN[f"log_mid_sum_fwd_{k_fwd}"] = (
            s.shift(-1).rolling(window=k_fwd, min_periods=k_fwd).sum().shift(-(k_fwd-1))
        )
    
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"]
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (dfN[f"log_mid_sum_fwd_{k_fwd}"] - bwd) / bwd.replace(0, np.nan)

    
        # --- classification indicator (avoid divide-by-zero) ---
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"].to_numpy()
        fwd = dfN[f"log_mid_sum_fwd_{k_fwd}"].to_numpy()
    
        denom = np.where(np.abs(bwd) < 1e-12, np.nan, bwd)
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (fwd - bwd) / denom
        dfN[f"log_mid_sum_ratio_{k_fwd}"].replace( np.nan,0)
    # ------------------------------------------------------------------------
    
    # --- imbalance / microprice / spread features on RESAMPLED series ---
    # 1. Protect against division by zero
    den = (dfN["bid_sz_00"] + dfN["ask_sz_00"]).replace(0, np.nan)
    
    # 2. Use a distinct name for the micro-price interpolation weight (0 to 1)
    dfN["bid_ratio_L1"] = dfN["bid_sz_00"] / den
    
    # 3. Calculate micro-price using the ratio
    dfN["micro_price"] = dfN["bid_ratio_L1"] * dfN["ask_px_00"] + (1.0 - dfN["bid_ratio_L1"]) * dfN["bid_px_00"]
    
    # 4. Calculate the standard Order Book Imbalance (OBI) metric (-1 to 1) for current model
    dfN["obi_L1"] = (dfN["bid_sz_00"] - dfN["ask_sz_00"]) / den
    dfN["obi_L1"] = dfN["obi_L1"].fillna(0.0) # Handle empty book cases safely

    dfN["imbalance"] = (dfN["bid_sz_00"] - dfN["ask_sz_00"]) / den
    dfN["imbalance"] = dfN["imbalance"].fillna(0.0)  # Neutral fallback for empty books



    dfN["spread"] = dfN["ask_px_00"] - dfN["bid_px_00"]
    dfN["rel_spread"] = dfN["spread"] / dfN["mid_price"]  # = 2*spread/(ask+bid)

    eps = 1e-12
    dfN["log_spread"] = np.log(dfN["ask_px_00"].clip(lower=eps)) - np.log(dfN["bid_px_00"].clip(lower=eps))

    dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
    dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])


    # --- “mid crosses previous spread” on RESAMPLED series ---
    prev_bid = dfN["bid_px_00"].shift(1)
    prev_ask = dfN["ask_px_00"].shift(1)

    dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
    dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)

    # --- “mid jump relative to spread” (resampled) ---
    dfN["log_mid_ret"] = dfN["log_mid"].diff()
    dfN["mid_ret"]     = dfN["mid_price"].diff()
    prev_spread = dfN["spread"].shift(1)
    
    dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)

    # Clean helper column if created
    if "_ts_event_dt" in dfN.columns:
        # keep it if you like; otherwise drop
        pass

    return dfN
#------------------------------------------------------------------------------
#-----VOLUME TIME RESAMPLING
#------------------------------------------------------------------------------
def bucket_sum(values, end_pos, prev_end_pos):
    x = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    cs = np.cumsum(x)

    prev = np.zeros(len(end_pos), dtype=float)
    ok = prev_end_pos >= 0
    prev[ok] = cs[prev_end_pos[ok]]

    return cs[end_pos] - prev

#------------------------------------------------------------------------------
def add_event_features_and_resample_volume(
    df: pd.DataFrame,
    n_shares: int = 10000,                   # volume resampling window
    W: int = 300,                         # back window for volatility estimATION
    k_fwdL = [1,2,3],                     # forward  windows for the predicted feature
    sort_cols=("ts_event", "sequence"),
    take: str = "last",                   # "last" keeps 10th/20th/... ; "first" keeps 1st/11th/...
    # VPIN params
    compute_vpin: bool = True,
    vpin_bucket_vol: float = 10000, #50_000.0,    # volume bucket size (shares/contracts)
    vpin_mavg: int = 10 # 20, 50,                  # rolling avg over last m buckets
) -> pd.DataFrame:

    #ofi_window = n         # For scale-invariance / fractal-style analysis
    ofi_window = 10        # For predictive modeling  
    n =   ofi_window
    
    if take not in {"last", "first"}:
        raise ValueError("take must be 'last' or 'first'")

    df2 = df.sort_values(list(sort_cols)).copy()
    tcol = "ts_event"  # tz-aware UTC datetime per Databento


    # --- forward-fill LOB snapshot columns (so every row has a complete state) ---
    lob_prefixes = ("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_")
    lob_cols = [c for c in df2.columns if c.startswith(lob_prefixes)]
    df2[lob_cols] = df2[lob_cols].ffill()

    # --- mid / log-mid / event returns (raw event series) ---
    df2["mid_price"] = (df2["bid_px_00"].astype("float64") + df2["ask_px_00"].astype("float64")) / 2.0
    df2["log_mid"] = np.log(df2["mid_price"].where(df2["mid_price"] > 0))
    df2["log_mid_ret"] = df2["log_mid"].diff()

    # --- event-volatility sigma_W at every raw event (RMS over last W event-returns) ---
    df2["sigma_W"] = (
        df2["log_mid_ret"]
           .rolling(W, min_periods=W)
           .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True)
    )
    

    # --- block features over last n raw events (aligned to each raw event) ---
    # Mid-price changes count over last n raw events
    dmid = df2["mid_price"].diff()
    df2["mid_changed"] = dmid.fillna(0).ne(0).astype(np.int8)
    
#------------------------------------------------------------------------------
    # --- L1 OFI (Cont-style) per event, then rolling sum over last n events ---
    bpx = df2["bid_px_00"].astype(float)
    apx = df2["ask_px_00"].astype(float)
    bsz = df2["bid_sz_00"].astype(float)
    asz = df2["ask_sz_00"].astype(float)

    bpx_prev, apx_prev = bpx.shift(1), apx.shift(1)
    bsz_prev, asz_prev = bsz.shift(1), asz.shift(1)

    db = np.where(bpx > bpx_prev,  bsz,
         np.where(bpx < bpx_prev, -bsz_prev,
                  bsz - bsz_prev))
    da = np.where(apx < apx_prev,  asz,
         np.where(apx > apx_prev, -asz_prev,
                  asz - asz_prev))
    df2["ofi_L1"] = (db - da)
    # order flow imbalance: Flow (LOB): ofi_L1_n (rolling OFI over last n events)
    df2["ofi_L1_n"] = pd.Series(df2["ofi_L1"], index=df2.index).rolling(n, min_periods=n).sum()

#------------------------------------------------------------------------------
    # Physical duration of last n events: τ_t - τ_{t-n+1} 
    # Using diff(periods=n-1) gives τ_t - τ_{t-(n-1)}
    # duration/intensity over last n events
    eps_time = 1e-9
    df2["dt_block_sec"] = df2[tcol].diff(n - 1).dt.total_seconds()
    df2["dt_block_sec"] = df2["dt_block_sec"].clip(lower=eps_time)

    df2["log_dt_block"] = np.log(df2["dt_block_sec"])
    df2["event_intensity"] = n / df2["dt_block_sec"]
    df2["log_event_intensity"] = np.log(df2["event_intensity"])
    #df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"]

#------------------------------------------------------------------------------
    # --- Trade-based rolling OFI over last n events (uses action/side/size) ---
    a = df2["action"].astype(str).str.upper()
    side = df2["side"].astype(str).str.upper()
    is_trade = a.str.startswith("T")  # adjust if your trade action differs
    size = df2["size"].astype(float).to_numpy()
    price = df2["price"].astype(float).to_numpy()
    mid = df2["mid_price"].to_numpy()
    
    # aggressor sign: +1 buy, -1 sell
    sgn = np.zeros(len(df2), dtype=np.int8)
    sgn[is_trade.values & (side.values == "B")] = 1
    sgn[is_trade.values & (side.values == "A")] = -1

    # fallback for side == 'N' on trade rows: midpoint/tick rule
    unk = is_trade.values & (side.values == "N")
    if np.any(unk):
        s = np.sign(price[unk] - mid[unk]).astype(np.int8)
        tick = np.sign(np.diff(price, prepend=price[0]))[unk].astype(np.int8)
        s = np.where(s != 0, s, tick)
        # carry last nonzero within the *trade stream* (simple pass)
        last = 0
        for i, val in enumerate(s):
            if val == 0:
                s[i] = last
            else:
                last = val
        sgn[unk] = s

    buy_vol_event = np.where(is_trade.values & (sgn > 0), size, 0.0)
    sell_vol_event = np.where(is_trade.values & (sgn < 0), size, 0.0)

    df2["trade_buy_vol_event"] = buy_vol_event
    df2["trade_sell_vol_event"] = sell_vol_event

    df2["trade_buy_vol_n"] = pd.Series(buy_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    df2["trade_sell_vol_n"] = pd.Series(sell_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    tv = df2["trade_buy_vol_n"] + df2["trade_sell_vol_n"]
    # Flow (trades): trade_ofi_n (rolling signed trade volume imbalance over last n events)
    df2["trade_ofi_n"] = (df2["trade_buy_vol_n"] - df2["trade_sell_vol_n"]) / tv.replace(0, np.nan)

    # --- VPIN (volume-bucket VPIN on trades), forward-filled to events ---
    df2["vpin"] = np.nan
    if compute_vpin:
        tr_idx = np.flatnonzero(is_trade.values)
        if tr_idx.size > 0:
            tr_size = size[tr_idx]
            tr_buy = buy_vol_event[tr_idx]
            tr_sell = sell_vol_event[tr_idx]

            cumv = np.cumsum(tr_size)
            bucket_id = (cumv // float(vpin_bucket_vol)).astype(np.int64)

            buy_b = pd.Series(tr_buy).groupby(bucket_id).sum()
            sell_b = pd.Series(tr_sell).groupby(bucket_id).sum()

            imb = (buy_b - sell_b).abs()
            vpin_bucket = (imb / float(vpin_bucket_vol)).rolling(vpin_mavg, min_periods=vpin_mavg).mean()

            # map bucket VPIN back to each trade, then ffill to all events
            vpin_trade = pd.Series(bucket_id).map(vpin_bucket).to_numpy()
            #df2.loc[df2.index[tr_idx], "vpin"] = vpin_trade
            col = df2.columns.get_loc("vpin")
            df2.iloc[tr_idx, col] = vpin_trade
            df2["vpin"] = df2["vpin"].ffill()
            df2["vpin"] = df2["vpin"].bfill()
            df2["dvpin"]= df2["vpin"].diff()
            df2["dvpin"]= df2["dvpin"].bfill()
#------------------------------------------------------------------------------
    # Optional: mid-change rate per second
    # df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"].clip(lower=eps_time)



     # df2 = add_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")
    #deep_ofi = df2["ofi_L10_n"]


    # df2 = add_normalized_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")
    # main feature:
    # deep_nofi_n = df2["ofi_L10_norm_n"]
    


   
    # --- volume-time resampling: keep the event where cumulative volume crosses j*n_shares ---
    
    trade_vol_event = df2["trade_buy_vol_event"] + df2["trade_sell_vol_event"]
    cumv = trade_vol_event.cumsum().to_numpy()
    
    thresholds = np.arange(n_shares, cumv[-1] + 1e-12, n_shares)
    
    end_pos = np.searchsorted(cumv, thresholds, side="left")
    
    valid = end_pos < len(df2)
    end_pos = end_pos[valid]
    thresholds = thresholds[valid]
    
    dfN = df2.iloc[end_pos].copy()
    
    # If df2 is already sorted by ts_event/sequence, this is already chronological.
    # Use sort_index only if the index is truly a timestamp index.
    dfN = dfN.sort_index()
    
    dfN["event_idx"] = end_pos
    dfN["sample_idx"] = np.arange(len(dfN), dtype=np.int64)
    dfN["cum_trade_vol"] = cumv[end_pos]
    dfN["volume_threshold"] = thresholds
    
    prev_end_pos = np.r_[-1, end_pos[:-1]].astype(np.int64)

    dfN["mid_change_count_vol_bucket"] = bucket_sum(
        df2["mid_changed"].to_numpy(),
        end_pos,
        prev_end_pos
    ).astype(np.int32)
    
    dfN["bucket_trade_vol"] = bucket_sum(
        trade_vol_event,
        end_pos,
        prev_end_pos
    )
    
    dfN["events_in_bucket"] = end_pos - prev_end_pos
    
    dfN["move_rate_per_share"] = (
        dfN["mid_change_count_vol_bucket"]
        / dfN["bucket_trade_vol"].replace(0, np.nan)
    )
    
    dfN["move_rate_per_event_in_vol_bucket"] = (
        dfN["mid_change_count_vol_bucket"]
        / dfN["events_in_bucket"].replace(0, np.nan)
    )


    
    
    # -------------------------------------------------------------------------
    
    
    # k-step-ahead log return on the resampled series 
    # --- forward k-step return on the RESAMPLED series (optional) ---
    for k_fwd in k_fwdL:
        s = dfN["log_mid"]
    
        # forward k-step return: log_mid[t+k] - log_mid[t]
        dfN[f"log_mid_return_fwd_{k_fwd}"] = s.shift(-k_fwd) - s
        dfN[f"log_mid_return_fwd_{k_fwd}"].replace( np.nan,0)
        # backward sum: log_mid[t-k+1] + ... + log_mid[t]
        dfN[f"log_mid_sum_bwd_{k_fwd}"] = s.rolling(window=k_fwd, min_periods=k_fwd).sum()
    
        # forward sum: log_mid[t+1] + ... + log_mid[t+k]
        # rolling is backward-looking => shift result by -(k-1)
        dfN[f"log_mid_sum_fwd_{k_fwd}"] = (
            s.shift(-1).rolling(window=k_fwd, min_periods=k_fwd).sum().shift(-(k_fwd-1))
        )
    
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"]
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (dfN[f"log_mid_sum_fwd_{k_fwd}"] - bwd) / bwd.replace(0, np.nan)

    
        # --- classification indicator (avoid divide-by-zero) ---
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"].to_numpy()
        fwd = dfN[f"log_mid_sum_fwd_{k_fwd}"].to_numpy()
    
        denom = np.where(np.abs(bwd) < 1e-12, np.nan, bwd)
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (fwd - bwd) / denom
        dfN[f"log_mid_sum_ratio_{k_fwd}"].replace( np.nan,0)
    # ------------------------------------------------------------------------
    
    # --- imbalance / microprice / spread features on RESAMPLED series ---
    den = (dfN["bid_sz_00"] + dfN["ask_sz_00"]).replace(0, np.nan)
    dfN["imbalance"] = dfN["bid_sz_00"] / den

    dfN["micro_price"] = dfN["imbalance"] * dfN["ask_px_00"] + (1.0 - dfN["imbalance"]) * dfN["bid_px_00"]

    dfN["spread"] = dfN["ask_px_00"] - dfN["bid_px_00"]
    dfN["rel_spread"] = dfN["spread"] / dfN["mid_price"]  # = 2*spread/(ask+bid)

    eps = 1e-12
    dfN["log_spread"] = np.log(dfN["ask_px_00"].clip(lower=eps)) - np.log(dfN["bid_px_00"].clip(lower=eps))

    dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
    dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])


    # --- “mid crosses previous spread” on RESAMPLED series ---
    prev_bid = dfN["bid_px_00"].shift(1)
    prev_ask = dfN["ask_px_00"].shift(1)

    dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
    dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)

    # --- “mid jump relative to spread” (resampled) ---
    dfN["log_mid_ret"] = dfN["log_mid"].diff()
    dfN["mid_ret"]     = dfN["mid_price"].diff()
    prev_spread = dfN["spread"].shift(1)
    
    dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)

    # Clean helper column if created
    if "_ts_event_dt" in dfN.columns:
        # keep it if you like; otherwise drop
        pass

    return dfN

def restrict_to_rth_et(df, ts_col_utc, start, end):
    # ts_event is tz-aware UTC per Databento
    df['ts_et'] = df[ts_col_utc].dt.tz_convert("America/New_York") 

    # Apply the filter
    df  = df[(df['ts_et'].dt.time >= start) & (df['ts_et'].dt.time <= end)]

    return df
def generate_timeseries(date, tStart,tEnd, frequency, k_fwdL, W, fName, fPath):
    # frequency -resampling frequency in events
    
    print('===================================================================')
    #print('START GENERATING TRAINING DATA FOR ',feature," AT ",frequency,'sec ON ',date)
    
    # future returns aligned with horizon 𝑘 - in trhe resampling frequency;
    # i.e.k*frequency events ahead

    # Output at frequency sampling:
     #'mid_price', 
     #'log_mid', 
     #log_mid_ret', 
     #'sigma_W', 
     #'mid_changed', - at evednt level->disregard
     #'mid_change_count_n', - number of midprice chages in the block - 0 ..n
     #'move_rate_per_event', - normalized number of chages per event
     #'dt_block_sec',        -- physical seconds per block (how long did it take to arrive and process n events) - 0.0 - 500   
     #'log_dt_block',        -- log(dt_block_sec) 
     #'event_intensity',     -- events per second 
     #'log_event_intensity', 
     #'move_rate_per_sec',   -- mid-price changes per second  
     #'event_idx_in_group',
     #'sample_idx', 
     #'log_mid_return_fwd_1',  -- k-step (1 block)-ahead log return on the resampled series
     #'log_mid_return_fwd_2'
     #'log_mid_return_fwd_3'
     #'imbalance',             -- dfN["bid_sz_00"]/(dfN["bid_sz_00"] + dfN["ask_sz_00"])
     #'micro_price',           -- dfN["imbalance"] * dfN["ask_px_00"] + (1.0 - dfN["imbalance"]) * dfN["bid_px_00"]
     #'spread',                -- dfN["ask_px_00"] - dfN["bid_px_00"]
     #'rel_spread', 
     #'log_spread', 
     #'micro_mid',             -- dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
     #'micro_mid_sign',        -- dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])
     #'mid_cross_prev_ask_up', -- dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
     #'mid_cross_prev_bid_dn', -- dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)
     #'mid_ret',               -- dfN["mid_ret"]     = dfN.groupby(group_col, sort=False)["mid_price"].diff()
     #'jump_gt_prev_spread'    -- dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)
     
     # ofi_L10: Weighted deep order-flow imbalance at the event level, combining OFI from book levels 0–9 (L1–L10) 
     # ofi_L10_n: Rolling sum of ofi_L10 over the last n events (cumulative deep OFI over the recent window).
     
     # ofi_L10_norm: Depth-normalized deep OFI: ofi_L10 divided by the total bid+ask size across levels 0–9 at that event (scale-free “pressure per available depth”).
     # ofi_L10_norm_n: Rolling depth-normalized deep OFI over the last n events: (sum of ofi_L10 over the window) divided by (sum of total depth over the window). 
     
     
     
    group_col = "instrument_id"   # or "symbol"
    sort_cols=("ts_event", "sequence")  
    take = "last"  
    df = dbn_to_df(fName, fPath)
    
    # restrict to normal trading time 09:30 - 15:30
    time_stamm_column_utc="ts_event"
    df = restrict_to_rth_et(df, time_stamm_column_utc , tStart,tEnd)
    
    dfr = add_event_features_and_resample(
            df,
            frequency,  # events resampling window 
            W,          # back window for volatility estimation
            k_fwdL,
            sort_cols,
            take # "last" keeps 10th/20th/... ; "first" keeps 1st/11th/...
        ) 
    
    return dfr


#------------------------------------------------------------------------------
         
def generate_timeseries_time(date, tStart,tEnd, frequency, k_fwdL, W, fName, fPath):
    group_col = "instrument_id"   # or "symbol"
    sort_cols=("ts_event", "sequence")  
    take = "last"  
    df = dbn_to_df(fName, fPath)

    # restrict to normal trading time 09:30 - 15:30
    time_stamm_column_utc="ts_event"    
    df = restrict_to_rth_et(df, time_stamm_column_utc , tStart,tEnd)
    
     
    dfr = add_event_features_and_resample_time(
        df,
        frequency,             # bucket size in seconds
        W ,             # trailing event-vol window on raw events (optional)
        k_fwdL    ,             # forward horizon in *buckets*
        sort_cols=sort_cols)
    return dfr
#------------------------------------------------------------------------------
                               
def generate_timeseries_volume(date, tStart,tEnd, frequency, k_fwdL, W, fName, fPath):
    group_col = "instrument_id"   # or "symbol"
    sort_cols=("ts_event", "sequence")  
    take = "last"  
    df = dbn_to_df(fName, fPath)

    # restrict to normal trading time 09:30 - 15:30
    time_stamm_column_utc="ts_event"    
    df = restrict_to_rth_et(df, time_stamm_column_utc , tStart,tEnd)
    
     
    dfr = add_event_features_and_resample_volume(
        df,
        frequency,             # bucket size in seconds
        W ,             # trailing event-vol window on raw events (optional)
        k_fwdL    ,             # forward horizon in *buckets*
        sort_cols=sort_cols)
    return dfr    



def get_timeseries_by_date(symbol,fPath, date,resampling,frequency, frw_intervals, tStart,tEnd ):
    fName = 'xnas-itch-'+date+'.mbp-10.dbn.zst'
    
    if resampling == 'events':
        print('Timeseries in events ',frequency,'events')# seconds
        freq_units='evn'
        W = 3*frequency # backward window to calculate volatility
        time_series = generate_timeseries(date, tStart,tEnd, frequency,  frw_intervals, W, fName, fPath)
    elif resampling == 'seconds':
        print('Timeseries in seconds ',frequency,'sec')# seconds
        freq_units = 'sec'
        W = 3*frequency # backward window to calculate volatility
        time_series = generate_timeseries_time(date, tStart,tEnd, frequency,frw_intervals, W, fName, fPath)
    elif resampling == 'volume':
        print('Timeseries in shares ',frequency,'shr')# seconds
        freq_units = 'shr'
        W = 3*frequency # backward window to calculate volatility
        time_series = generate_timeseries_volume(date, tStart, tEnd, frequency,frw_intervals, W, fName, fPath)            
    return time_series

def get_bivariate_ts(time_series, predicted, predictor, alpha, n_symbols):
        #------------------------------------------------------------------------------
        #              z-encoding for BI Variate
        # Encoding with 4 symbols per variate
         
        alpha = 0.05
        time_series_2z = z_encoding(time_series, predictor, n_symbols,  alpha, fill_mode = "bfill", bins_mode = "expanding_quantile")
        time_series_2z = z_encoding(time_series, predicted, n_symbols,  alpha, fill_mode = "bfill", bins_mode = "expanding_quantile")

        predicted_series = time_series_2z[predicted+'_sym']
        predictor_series = time_series_2z[predictor+'_sym']
        
        #------------------------------------------------------------------
        z_series12 = pd.concat([predicted_series, predictor_series], axis=1)
        ni = np.asarray([n_symbols,n_symbols])
        weights = np.concatenate(([1], np.cumprod(ni[:-1])))
        z_series12 = (z_series12* weights).sum(axis=1)
          
        
        # bi_series = list( z_series12.astype(int))
        
        
        return z_series12.astype(int)

# -------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Z-encoding of tume series
#------------------------------------------------------------------------------
# ---------- Basic building blocks ----------
def ewma_zscore(series: pd.Series, alpha: float) -> pd.Series:
    x = series.astype(float)
    mu = x.ewm(alpha=alpha, adjust=False).mean()
    sq = (x - mu)**2
    var = sq.ewm(alpha=alpha, adjust=False).mean()
    sigma = np.sqrt(var)
    z = (x - mu) / sigma
    return z.replace([np.inf, -np.inf], np.nan).dropna()
#-----------------------------------------------------------------------------
def ewma_zscore_predictive(series: pd.Series, alpha: float, min_periods: int = 2,
                           eps: float = 1e-12) -> pd.Series:
    """
    Predictive (causal) EWMA z-score:
        z[i] = (x[i] - mu[i-1]) / sigma[i-1]
    where mu, sigma are computed from past only.
    """
    x = series.astype("float64")

    # Past-only series for estimating mu_{i-1}, sigma_{i-1}
    x_lag = x.shift(1)

    # EWMA mean of past
    mu_prev = x_lag.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()

    # EWMA variance of past (EWMA of squared deviations from past mean)
    dev_prev = x_lag - mu_prev
    var_prev = (dev_prev * dev_prev).ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
    sigma_prev = np.sqrt(var_prev).clip(lower=eps)

    z = (x - mu_prev) / sigma_prev
    z = z.replace([np.inf, -np.inf], np.nan)
    return z

#-----------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from statistics import NormalDist


def make_symmetric_z_bins(m: int, zmax: float = 3.0) -> np.ndarray:
    # evenly spaced thresholds in [-zmax, zmax]
    return np.linspace(-zmax, zmax, m + 1)[1:-1]

def make_normal_quantile_z_bins(m: int, clip: float = 3.0) -> np.ndarray:
    nd = NormalDist()
    qs = [i / m for i in range(1, m)]
    bins = np.array([nd.inv_cdf(q) for q in qs], dtype=float)
    return np.clip(bins, -clip, clip)

#--------------------------------------------------------------------------
# Support for predictive z-encoding - binning
def expanding_quantile_edges(
    z: pd.Series,
    n_symbols: int,
    bin_min_periods: int = 1,
) -> pd.DataFrame:
    """
    For each time t, compute quantile bin edges from z[:t-1].

    Returns a DataFrame with n_symbols-1 columns:
        edge_1, ..., edge_{n_symbols-1}
    """
    if n_symbols < 2:
        raise ValueError("n_symbols must be >= 2")

    z_hist = z.shift(1)  # important: only past z-values are used

    qs = np.arange(1, n_symbols) / n_symbols

    edges = []
    for q in qs:
        e = z_hist.expanding(min_periods=bin_min_periods).quantile(q)
        edges.append(e)

    edges = pd.concat(edges, axis=1)
    edges.columns = [f"edge_{j}" for j in range(1, n_symbols)]

    return edges


def discretize_z_timevarying_bins(
    z: pd.Series,
    edges: pd.DataFrame,
    missing_symbol: int = -1,
) -> pd.Series:
    """
    Discretize z_t using time-varying bin edges at time t.

    Symbol is number of edges less than or equal to z_t.
    Produces symbols 0, ..., n_symbols-1.
    """
    z_arr = z.to_numpy(dtype=float)
    E = edges.to_numpy(dtype=float)

    sym = np.full(len(z_arr), missing_symbol, dtype=np.int32)

    valid = np.isfinite(z_arr) & np.all(np.isfinite(E), axis=1)

    if valid.any():
        # count how many thresholds z_t crosses
        sym[valid] = np.sum(z_arr[valid, None] >= E[valid], axis=1).astype(np.int32)

    return pd.Series(sym, index=z.index)

# -------------------------------------------------------------------------
"""
bins_mode options
-----------------

1) bins_mode="daily_quantile"

   Fits symbol thresholds from the empirical quantiles of the predictive
   z-scores over the entire provided series.

   Interpretation:
       Offline diagnostic encoding.

   Leakage status:
       Not strictly predictive if the same interval is used for evaluation,
       because the bin edges use future z-values.

   Typical use:
       Exploratory MI/TE diagnostics, full-day distribution studies.


2) bins_mode="prefix_quantile"

   Fits symbol thresholds from the empirical quantiles of the predictive
   z-scores over the first fit_frac fraction of the provided series.

   Interpretation:
       The initial prefix is treated as a calibration / burn-in region.

   Leakage status:
       Predictive only after the prefix. The prefix itself should not be
       treated as out-of-sample, because bin edges are fit using the full
       prefix.

   Typical use:
       Single-session experiments where the beginning of the day is used
       to calibrate symbolic bins before forecasting later periods.


3) bins_mode="history_quantile"

   Fits symbol thresholds from all observations with timestamp <= fit_until.
   The timestamp is taken either from time_col or from the DataFrame index.

   Interpretation:
       Uses all historically available data before the prediction-start time.

   Leakage status:
       Predictive for observations after fit_until, provided no later data
       are included in the calibration set.

   Typical use:
       Production-style or walk-forward experiments:
           previous sessions + current observed prefix -> fit bins
           future prediction interval                  -> encode/evaluate


4) bins_mode="fixed_uniform"

   Uses fixed z-score thresholds evenly spaced in [-zmax, zmax].

   Example:
       n_symbols=8, zmax=3 gives thresholds roughly
       [-2.25, -1.50, -0.75, 0.00, 0.75, 1.50, 2.25].

   Interpretation:
       Symbols represent fixed deviations from the predictive EWMA mean
       in units of predictive EWMA standard deviation.

   Leakage status:
       Fully predictive. No bin fitting is performed.

   Typical use:
       Strict online encoding, robust cross-day comparison, symbolic
       generative modeling where interpretability is preferred over balanced
       symbol frequencies.


5) bins_mode="fixed_normal"

   Uses fixed thresholds given by standard-normal quantiles, optionally
   clipped to [-zmax, zmax].

   Interpretation:
       Similar to fixed_uniform, but bins are closer to equal-probability
       under an approximately Gaussian z-score distribution.

   Leakage status:
       Fully predictive. No bin fitting is performed.

   Typical use:
       Default recommended fixed-bin scheme when one wants no lookahead but
       better occupancy balance than uniformly spaced z-thresholds.
"""
# 
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
def z_encoding(
    time_series: pd.DataFrame,
    feature: str,
    n_symbols: int,
    alpha: float,
    # optional params
    min_periods: int = 2,
    bins_mode: str = "expanding_quantile",
    fit_frac: float = 0.5,
    fit_until=None,
    time_col: str | None = None,
    zmax: float = 3.0,
    fill_mode: str = "ffill",
    missing_symbol: int = -1,
    bin_min_periods: int = 1,   # new: for expanding_quantile
) -> pd.DataFrame:

    x = time_series[feature]

    # predictive z-score: z_t uses only x_{<t}
    z = ewma_zscore_predictive(x, alpha=alpha, min_periods=min_periods).bfill()
    #z.isna().sum()
    # ------------------------------------------------------------------
    # fixed-bin cases
    # ------------------------------------------------------------------
    if bins_mode == "daily_quantile":
        bins = fit_quantile_bins(z, n_symbols)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "prefix_quantile":
        T = max(int(len(z) * fit_frac), 1)
        bins = fit_quantile_bins(z.iloc[:T], n_symbols)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "history_quantile":
        if fit_until is None:
            raise ValueError("fit_until must be provided for bins_mode='history_quantile'.")

        if time_col is None:
            ts = time_series.index
        else:
            ts = pd.to_datetime(time_series[time_col])

        fit_mask = ts <= fit_until

        if fit_mask.sum() < n_symbols:
            raise ValueError("Not enough observations before fit_until.")

        bins = fit_quantile_bins(z.loc[fit_mask], n_symbols)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "fixed_uniform":
        bins = make_symmetric_z_bins(n_symbols, zmax=zmax)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "fixed_normal":
        bins = make_normal_quantile_z_bins(n_symbols, clip=zmax)
        s = discretize_z(z, bins=bins)

    # ------------------------------------------------------------------
    # fully online case: each t uses bins fitted on z_{<t}
    # ------------------------------------------------------------------
    elif bins_mode == "expanding_quantile":
        edges = expanding_quantile_edges(
            z,
            n_symbols=n_symbols,
            bin_min_periods=bin_min_periods,
        )
        s = discretize_z_timevarying_bins(
            z,
            edges,
            missing_symbol=missing_symbol,
        )

    else:
        raise ValueError(
            "bins_mode must be one of: "
            "daily_quantile, prefix_quantile, history_quantile, "
            "fixed_uniform, fixed_normal, expanding_quantile"
        )

    # ------------------------------------------------------------------
    # fill handling
    # ------------------------------------------------------------------
    if fill_mode == "ffill":
        z = z.ffill()
        s = s.ffill().fillna(missing_symbol).astype(np.int32)

    elif fill_mode == "bfill_ffill":
        # non-causal; use only for offline diagnostics
        z = z.bfill().ffill()
        s = s.bfill().ffill().astype(np.int32)

    elif fill_mode == "bfill":
        # non-causal; use only for offline diagnostics
        s = s.replace(missing_symbol, np.nan).bfill().astype("int")
    elif fill_mode == "none":
        s = s.where(z.notna(), other=missing_symbol).astype("int")

    else:
        raise ValueError("fill_mode must be: ffill, bfill_ffill, none")

    time_series[feature + "_z"] = z
    time_series[feature + "_sym"] = s

    return time_series

JointKey = Tuple[Tuple[int, ...], ...]
ComponentKey = Tuple[int, ...]


def _finalize_table(
    occurrence_counts: Counter,
    class_counts: Dict[Any, np.ndarray],
    n_classes: int,
    smoothing: float = 0.0,
) -> Dict[str, Any]:
    """
    Convert count dictionaries into arrays.

    occurrence_counts[key]:
        Total number of occurrences of the sequence/key.

    class_counts[key][c]:
        Number of occurrences associated with class c.
    """
    keys = list(occurrence_counts.keys())

    if not keys:
        raise ValueError("No valid sequence samples were generated.")

    counts = np.asarray(
        [occurrence_counts[key] for key in keys],
        dtype=np.int64,
    )

    target_counts = np.stack(
        [class_counts[key] for key in keys],
        axis=0,
    ).astype(np.int64)

    if not np.array_equal(target_counts.sum(axis=1), counts):
        raise RuntimeError(
            "Class counts do not sum to the sequence occurrence counts."
        )

    sequence_probabilities = counts / counts.sum()

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    target_distributions = (
        target_counts + smoothing
    ) / (
        counts[:, None] + smoothing * n_classes
    )

    return {
        "keys": keys,
        "index": {key: j for j, key in enumerate(keys)},
        "counts": counts,
        "class_counts": target_counts,
        "seq_probs": sequence_probabilities.astype(np.float64),
        "target_distributions": target_distributions.astype(np.float64),
        "n_occurrences": int(counts.sum()),
    }

def add_class_label(
    time_series: pd.DataFrame,
    class_name: str,
    fill_missing: bool = False,
) -> pd.Series:
    """
    Add a {-1,0,1} classification column to time_series and return it.

    Supported class_name:
      c{k}   : based on log_mid_return_fwd_k
      ca{k}  : based on log_mid_sum_fwd_k vs log_mid_sum_bwd_k

    Examples:
      add_class_label(time_series, "c1",  theta=0.0000415)
      add_class_label(time_series, "c2",  theta=0.0000415)
      add_class_label(time_series, "c4",  theta=0.000075)
      add_class_label(time_series, "ca1", theta=0.000007)
      add_class_label(time_series, "ca2", theta=0.000007)
      add_class_label(time_series, "ca4", theta=0.000007)
    """
    thetaDict = {}
    thetaDict["c1"] =0.0000415
    thetaDict["c2"] =0.0000415
    thetaDict["c4"] =0.000075
    thetaDict["ca1"]=0.000007
    thetaDict["ca2"]=0.000007
    thetaDict["ca4"]=0.000007
    
    theta = thetaDict[class_name]
    if class_name.startswith("ca"):
        k = int(class_name[2:])

        fwd_col = f"log_mid_sum_fwd_{k}"
        bwd_col = f"log_mid_sum_bwd_{k}"

        if fwd_col not in time_series.columns:
            raise KeyError(f"Missing column: {fwd_col}")
        if bwd_col not in time_series.columns:
            raise KeyError(f"Missing column: {bwd_col}")

        fwd = time_series[fwd_col]
        bwd = time_series[bwd_col]

        if fill_missing:
            fwd = fwd.ffill()
            bwd = bwd.bfill()

        # reproduce your original scaling:
        # ca1 -> theta
        # ca2 -> 1.5 * theta
        # ca4 -> 2.0 * theta
        theta_mult = {
            1: 1.0,
            2: 1.5,
            4: 2.0,
        }.get(k, 1.0)

        th = theta * theta_mult

        time_series[class_name] = np.select(
            [
                fwd > bwd * (1.0 + th),
                fwd < bwd * (1.0 - th),
            ],
            [1, -1],
            default=0,
        ).astype(np.int8)

    elif class_name.startswith("c"):
        k = int(class_name[1:])

        ret_col = f"log_mid_return_fwd_{k}"

        if ret_col not in time_series.columns:
            raise KeyError(f"Missing column: {ret_col}")

        r = time_series[ret_col]

        if fill_missing:
            r = r.ffill()

        time_series[class_name] = np.select(
            [
                r > theta,
                r < -theta,
            ],
            [1, -1],
            default=0,
        ).astype(np.int8)

    else:
        raise ValueError("class_name must start with 'c' or 'ca', e.g. 'c1', 'c2', 'ca1', 'ca4'.")

    return time_series[class_name]



def prepare_ensemble_training_data(
    daily_data: Sequence[Tuple[Sequence[pd.Series], pd.Series]],
    sequence_length: int,
    class_values: Sequence[int] = (-1, 0, 1),
    alphabet_size: int = 16,
    smoothing: float = 0.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Build joint ensemble data and marginal component data.

    Parameters
    ----------
    daily_data
        One element per day:

            (
                [z_series_1, ..., z_series_n],
                class_series
            )

        All objects must be pandas Series indexed by timestamp.

    sequence_length
        Fixed sequence length k.

    class_values
        Ordered class labels. For example (-1, 0, 1).

    alphabet_size
        Alphabet size of each bivariate encoding. With two four-symbol
        features this is 4 * 4 = 16.

    smoothing
        Optional symmetric Dirichlet pseudocount for target class
        distributions. Use 0.0 for raw empirical distributions.

    Returns
    -------
    joint_data
        Unique joint sequences and their empirical class distributions.

        joint_data["X"] has shape:
            [n_unique_joint_sequences, n_channels, sequence_length]

    component_data
        List of length n_channels.

        component_data[i]["X"] has shape:
            [n_unique_component_sequences_i, sequence_length]
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    if not daily_data:
        raise ValueError("daily_data is empty.")

    class_to_index = {
        int(label): idx for idx, label in enumerate(class_values)
    }
    n_classes = len(class_values)

    n_channels = len(daily_data[0][0])
    if n_channels < 1:
        raise ValueError("At least one channel is required.")

    # Joint ensemble counts.
    joint_occurrence_counts: Counter[JointKey] = Counter()
    joint_class_counts = defaultdict(
        lambda: np.zeros(n_classes, dtype=np.int64)
    )

    # Marginal component counts.
    component_occurrence_counts = [
        Counter() for _ in range(n_channels)
    ]
    component_class_counts = [
        defaultdict(lambda: np.zeros(n_classes, dtype=np.int64))
        for _ in range(n_channels)
    ]

    daily_sample_counts: List[int] = []

    for day_index, (z_series_list, class_series) in enumerate(daily_data):
        if len(z_series_list) != n_channels:
            raise ValueError(
                f"Day {day_index} has {len(z_series_list)} channels; "
                f"expected {n_channels}."
            )

        # Inner join guarantees exact timestamp alignment.
        aligned = pd.concat(
            [
                series.rename(f"z_{i}")
                for i, series in enumerate(z_series_list)
            ]
            + [class_series.rename("class")],
            axis=1,
            join="inner",
        )

        aligned = (
            aligned
            .sort_index()
            .loc[lambda x: ~x.index.duplicated(keep="last")]
            .dropna()
        )

        z_columns = [f"z_{i}" for i in range(n_channels)]

        Z = aligned[z_columns].to_numpy(dtype=np.int64)
        labels = aligned["class"].to_numpy(dtype=np.int64)

        if np.any(Z < 0) or np.any(Z >= alphabet_size):
            bad = Z[(Z < 0) | (Z >= alphabet_size)]
            raise ValueError(
                f"Day {day_index} contains symbols outside "
                f"[0, {alphabet_size - 1}]. Examples: {bad[:10]}"
            )

        if len(Z) < sequence_length:
            daily_sample_counts.append(0)
            continue

        valid_samples = 0

        # Process this day independently, so sequences never cross
        # overnight/day boundaries.
        for end in range(sequence_length - 1, len(Z)):
            class_label = int(labels[end])

            # Skip undefined or unsupported labels.
            if class_label not in class_to_index:
                continue

            class_index = class_to_index[class_label]

            # Shape: [sequence_length, n_channels].
            window = Z[
                end - sequence_length + 1 : end + 1,
                :
            ]

            # Store the joint key as n channel-specific sequences.
            #
            # joint_key[i] =
            #     sequence observed by encoder i.
            joint_key: JointKey = tuple(
                tuple(int(value) for value in window[:, i])
                for i in range(n_channels)
            )

            joint_occurrence_counts[joint_key] += 1
            joint_class_counts[joint_key][class_index] += 1

            # Simultaneously construct the marginal component tables.
            for i, component_key in enumerate(joint_key):
                component_occurrence_counts[i][component_key] += 1
                component_class_counts[i][component_key][class_index] += 1

            valid_samples += 1

        daily_sample_counts.append(valid_samples)

    # Joint table.
    joint_data = _finalize_table(
        joint_occurrence_counts,
        joint_class_counts,
        n_classes=n_classes,
        smoothing=smoothing,
    )

    # Shape: [N_joint, n_channels, sequence_length].
    joint_data["X"] = np.asarray(
        joint_data["keys"],
        dtype=np.int16,
    )
    joint_data["class_values"] = tuple(class_values)
    joint_data["sequence_length"] = sequence_length
    joint_data["n_channels"] = n_channels
    joint_data["daily_sample_counts"] = np.asarray(
        daily_sample_counts,
        dtype=np.int64,
    )

    # Component tables.
    component_data: List[Dict[str, Any]] = []

    for i in range(n_channels):
        table_i = _finalize_table(
            component_occurrence_counts[i],
            component_class_counts[i],
            n_classes=n_classes,
            smoothing=smoothing,
        )

        # Shape: [N_component_i, sequence_length].
        table_i["X"] = np.asarray(
            table_i["keys"],
            dtype=np.int16,
        )
        table_i["class_values"] = tuple(class_values)
        table_i["sequence_length"] = sequence_length
        table_i["channel_index"] = i

        component_data.append(table_i)

    return joint_data, component_data

###############################################################################
#  Data Preparation
###############################################################################
def run_reference_driver():
    """Original module-level driver, unchanged (colleague's scope:
    AAPL, Windows path). Previously executed on import; now only runs
    when this file is executed directly."""
    fPath = 'C:\\Vanio\\Projects\\Market Data Preparation\\Data' 
    symbol= 'AAPL'
    # "normal" eastern trading time
    tStart = datetime.time(9, 30)
    tEnd   = datetime.time(15, 30)
    resampling= 'events' # "volume",  "seconds" 
    frequency = 100      # events or 1000 shares or 10s 

    frw_intervals = [1,2,3,4] # steps ahead price to be added to the time sereis

    feature_list = ['log_mid',"micro_price", 'obi_L1',"ofi_L1_n", "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n','tvi_n' ]
    feature_list = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n_norm",'ofi_L3_norm_n','ofi_L10_norm_n',"ofi_L1_n", 'ofi_L1_norm_n',"micro_price"]
    feature_list = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n_norm"]
    feature_list = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n_norm",'ofi_L3_norm_n','ofi_L10_norm_n',"micro_price","ofi_L1_n", 'ofi_L1_norm_n']

    features     = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n", "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n',"micro_price",'vpin', 'sigma_W' ]

    alpha = 0.05             # parameter used by the z-encoding - EWMA  
    n_symbols=4              # symbolic encoding of one time sereis 

    alphabet_size = n_symbols*n_symbols
    alphabet = list(range(alphabet_size)) # combined bivariate 

    max_seq_length=6
    max_subsequence_length=max_seq_length


    #-----------------------------------------------------------------------------
    # Integration dates
    #-----------------------------------------------------------------------------
    dates = ['20250401', '20250402', '20250403', '20250404','20250407', '20250408', '20250409', '20250410',
             '20250411', '20250414', '20250415', '20250416','20250417', '20250421', '20250422', '20250423',
             '20250424', '20250425', '20250428', '20250429','20250430'
             ]

    predicted = features[0]              # every bi-variate ts is combination of predictor and predicted feature
    class_calculation = True             # calculate class distribution by sub-sequence 
    # class Names 'c1', 'c2','c4', 'ca1','ca2','ca4' # 1,2,4 steps ahead and aggregated steps ahead

    cls_Names = [ 'c1', 'c2','c4', 'ca2','ca4']
    num_class_values = 3            #every classification includes -1, 0, 1 down, nochange, up
    class_values=(-1, 0, 1)

    seq_lengths = [1,2,3,4,5]


    for seq_length in seq_lengths:
        for cls_name in [ 'c1', 'c2', 'c4', 'ca2','ca4']:
    
    
        
            #seq_length =seq_lengths[0]
        
        
            predicted = features[0]

    
            # -------------------------------------------------------------------------
            # Calclate for all features
            #
            predictors = features
            predictors_list = 'ALL_'+str(len(predictors))
            # 
            #--------------------------------------------------------------------------
        
        
            daily_data = []
        
            #################################################################
        
            ################################################################
            for date in dates:
                print("Training Date ", date, "Class", cls_name, 'Seq Length', seq_length)
                z_series_list = [ 
                    get_bivariate_ts(
                        get_timeseries_by_date(symbol,fPath, date, resampling,frequency,frw_intervals ,tStart,tEnd ),
                        predicted,
                        predictor,
                        alpha,
                        n_symbols,
                    )
                    for predictor in predictors
                ]
            
                cls_series = add_class_label(
                    get_timeseries_by_date(symbol,fPath, date, resampling,frequency,frw_intervals ,tStart,tEnd ),
                    cls_name, )
        
                daily_data.append(
                    (z_series_list, cls_series)
                )
        
        
        
            joint_data, component_data = prepare_ensemble_training_data(
                daily_data=daily_data,
                sequence_length=seq_length,
                class_values=class_values,
                alphabet_size=alphabet_size,
                smoothing=0.0,
            )
        
            ###############################################################################
            # SAVE in FILE
            save_file_name ='ENS_TD_'+symbol+'_'+dates[0][0:6]+'_SL_'+str(seq_length)+'_CL_'+cls_name+"_"+predicted+"_"+predictors_list
        
            pickle.dump( [joint_data, component_data], open( save_file_name , "wb") )
        
            print('Dumped ',save_file_name)
    ###############################################################################













if __name__ == "__main__":
    run_reference_driver()
