"""Export daily C2C + O2O returns (PERMNO x date) for local PEAD work.

Sources (both are the project's authoritative ones, no ad-hoc reconstruction):
  O2O_RET, MarketCap : yifei's O2O_RET_by_permno_with_future_ret_v5.pq
                       (1996-01-02 .. 2026-03-30; the y of the whole O2O
                        residual/factor line, so CAR will be口径-consistent)
  RET (C2C), prices  : CRSP daily pickles under factor_rebuild/Data/dsf
                       (1993-01-04 .. 2026-04-30, raw `ret` straight from CRSP)

Conventions
  O2O_RET[t] : open_{t-1} -> open_t   (backward, split/div adjusted)
  RET[t]     : close_{t-1} -> close_t (CRSP `ret`, includes dividends)
  v5's own `ret` column is NOT C2C -- it is the forward O2O (open_t -> open_{t+1}),
  i.e. O2O_RET shifted by -1; verified on 2019-01 (matches old O2O_RET+1 exactly
  on 97.8% of rows, the rest being the documented v4->v5 revision). So C2C has to
  come from CRSP directly.

Rows are the union of the two sources; O2O_RET/MarketCap are NaN outside v5's
coverage, RET is NaN for the (few) v5 rows CRSP has no record for.
"""
import glob
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

V5 = "/project/dachxiu/yifei/news_data/Cache/US/O2O_RET_by_permno_with_future_ret_v5.pq"
DSF_DIR = "/scratch/midway3/zan1/factor_rebuild/Data/dsf"
OUT_DIR = "/project/dachxiu/zan/PEAD/export"
TMP_DIR = "/project/dachxiu/zan/PEAD/export/_tmp_v5_by_year"
OUT = os.path.join(OUT_DIR, "crsp_daily_ret_c2c_o2o.parquet")

SCHEMA = pa.schema([
    ("PERMNO", pa.int32()),
    ("date", pa.date32()),
    ("RET", pa.float64()),        # C2C, CRSP
    ("O2O_RET", pa.float64()),    # open_{t-1} -> open_t, v5
    ("MarketCap", pa.float64()),  # v5
    ("PRC", pa.float32()),        # CRSP close, abs()
    ("OPENPRC", pa.float32()),    # CRSP open, abs()
    ("SHROUT", pa.float32()),
])


def split_v5_by_year():
    """Pass 1: stream the 16 GB v5 file into one small shard per year."""
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)
    f = pq.ParquetFile(V5)
    writers, n = {}, 0
    for b in f.iter_batches(batch_size=1_000_000,
                            columns=["DATE", "PERMNO", "O2O_RET", "MarketCap"]):
        d = b.to_pandas().reset_index()
        d = d[["DATE", "PERMNO", "O2O_RET", "MarketCap"]]
        d["PERMNO"] = d["PERMNO"].astype("int32")
        for year, part in d.groupby(d["DATE"].dt.year):
            t = pa.Table.from_pandas(part, preserve_index=False)
            if year not in writers:
                writers[year] = pq.ParquetWriter(
                    os.path.join(TMP_DIR, f"{year}.parquet"), t.schema, compression="zstd")
            writers[year].write_table(t)
        n += len(d)
        if n % 10_000_000 == 0:
            print(f"  v5 scanned {n:,}", flush=True)
    for w in writers.values():
        w.close()
    print(f"  v5 done: {n:,} rows -> {len(writers)} year shards", flush=True)
    return sorted(writers)


def load_dsf_year(year):
    files = sorted(glob.glob(os.path.join(DSF_DIR, f"{year}*.pkl")))
    if not files:
        return None
    d = pd.concat([pd.read_pickle(p)[["permno", "date", "openprc", "prc", "ret", "shrout"]]
                   for p in files], ignore_index=True)
    d["permno"] = d["permno"].astype("int32")
    d["openprc"] = d["openprc"].abs()
    d["prc"] = d["prc"].abs()          # CRSP: negative price = bid/ask midpoint
    return d.rename(columns={"permno": "PERMNO", "prc": "PRC", "openprc": "OPENPRC",
                             "ret": "RET", "shrout": "SHROUT"})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("pass 1: split v5 by year", flush=True)
    v5_years = split_v5_by_year()

    dsf_years = sorted({int(os.path.basename(p)[:4])
                        for p in glob.glob(os.path.join(DSF_DIR, "*.pkl"))})
    years = sorted(set(v5_years) | set(dsf_years))
    print(f"\npass 2: merge per year ({years[0]}..{years[-1]})", flush=True)

    writer = pq.ParquetWriter(OUT, SCHEMA, compression="zstd", compression_level=9)
    total = 0
    try:
        for year in years:
            shard = os.path.join(TMP_DIR, f"{year}.parquet")
            v5 = pd.read_parquet(shard) if os.path.exists(shard) else None
            dsf = load_dsf_year(year)
            if v5 is not None:
                v5 = v5.rename(columns={"DATE": "date"})
                v5["date"] = pd.to_datetime(v5["date"])
            if v5 is None:
                m = dsf
                m["O2O_RET"] = pd.NA
                m["MarketCap"] = pd.NA
            elif dsf is None:
                m = v5
                for c in ["RET", "PRC", "OPENPRC", "SHROUT"]:
                    m[c] = pd.NA
            else:
                m = dsf.merge(v5, on=["PERMNO", "date"], how="outer")
            m = m[["PERMNO", "date", "RET", "O2O_RET", "MarketCap", "PRC", "OPENPRC", "SHROUT"]]
            m = m.sort_values(["date", "PERMNO"])
            for c, t in [("RET", "float64"), ("O2O_RET", "float64"), ("MarketCap", "float64"),
                         ("PRC", "float32"), ("OPENPRC", "float32"), ("SHROUT", "float32")]:
                m[c] = pd.to_numeric(m[c], errors="coerce").astype(t)
            writer.write_table(pa.Table.from_pandas(m, schema=SCHEMA, preserve_index=False))
            total += len(m)
            print(f"  {year}: {len(m):>9,} rows  (C2C {m['RET'].notna().mean():.3f}, "
                  f"O2O {m['O2O_RET'].notna().mean():.3f})", flush=True)
    finally:
        writer.close()

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\nwrote {OUT}\n  rows: {total:,}\n  size: {os.path.getsize(OUT)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
