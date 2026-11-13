import argparse
import glob
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import pymysql


def _load_target_entry_vids() -> Set[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "data", "target_cache"),
        os.path.join(here, "..", "system", "data", "target_cache"),
    ]
    vids: Set[str] = set()
    for root in candidates:
        for fp in glob.glob(os.path.join(root, "target_ids_*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            for ids in (data.get("ids") or {}).values():
                for v in ids:
                    vid = sanitize_vid(str(v))
                    if vid:
                        vids.add(vid)
    return vids


DB_CONFIG = {
    "host": os.environ.get("STARROCKS_HOST", "<YOUR_STARROCKS_HOST>"),
    "port": int(os.environ.get("STARROCKS_PORT", "9030")),
    "user": os.environ.get("STARROCKS_USER", "<YOUR_STARROCKS_USER>"),
    "password": os.environ.get("STARROCKS_PASSWORD", "<YOUR_STARROCKS_PASSWORD>"),
    "database": os.environ.get("STARROCKS_DATABASE", "<YOUR_STARROCKS_DATABASE>"),
    "charset": "utf8mb4",
}


PREFIX_TO_OUTFILE: Dict[str, str] = {
    "wechat_pay_account": "wechat_pay_account_samples.json",
}
BLACK_LIMIT_PER_TABLE = 5000
WHITE_LIMIT_PER_TABLE = 10000
SUFFIX_RULES: List[Tuple[str, str, int]] = [
    (os.environ.get("WHITE_TABLE_SUFFIX", "<YOUR_WHITE_TABLE_SUFFIX>"), "white", WHITE_LIMIT_PER_TABLE),
    (os.environ.get("BLACK_TABLE_SUFFIX", "<YOUR_BLACK_TABLE_SUFFIX>"), "black", BLACK_LIMIT_PER_TABLE),
]


def build_suffix_rules(black_limit: int, white_limit: int) -> List[Tuple[str, str, int]]:
    return [(suf, tag, white_limit if tag == "white" else black_limit)
            for suf, tag, _lim in SUFFIX_RULES]


def apply_out_tag(outfile_name: str, tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return outfile_name
    stem, ext = os.path.splitext(outfile_name)
    return f"{stem}_{tag}{ext}"


VID_SIZE = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data")
N_BLACK_SAMPLE = 1000
N_WHITE_SAMPLE = 2000
RANDOM_SEED = 42


def sanitize_vid(vid) -> str:
    if vid is None:
        return ""
    s = str(vid)
    encoded = s.encode("utf-8")
    if len(encoded) <= VID_SIZE:
        return s
    return encoded[:VID_SIZE].decode("utf-8", errors="ignore")


def connect_starrocks() -> pymysql.connections.Connection:
    print(
        f"[OK] Connecting to StarRocks: {DB_CONFIG['host']}:{DB_CONFIG['port']}/"
        f"{DB_CONFIG['database']} ..."
    )
    conn = pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset=DB_CONFIG.get("charset", "utf8mb4"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    print("[OK] StarRocks connected")
    return conn


def has_target_id_column(conn, table: str) -> bool:
    sql = f"DESC `{table}`"
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall() or []
    except Exception as ex:
        print(f"    [WARN] DESC `{table}` failed; skipping: {ex}")
        return False
    for r in rows:
        for k in ("Field", "field", "FIELD"):
            if k in r:
                if str(r[k]).strip().lower() == "target_id":
                    return True
                break
    return False


def table_exists(conn, table: str) -> bool:
    sql = "SHOW TABLES LIKE %s"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (table,))
            rows = cur.fetchall() or []
        return len(rows) > 0
    except Exception as ex:
        print(f"    [WARN] SHOW TABLES LIKE `{table}` failed: {ex}")
        return False


def get_partition_dates(conn, table: str) -> Optional[List]:
    sql = (
        f"SELECT DISTINCT `partition_date` AS pd FROM `{table}` "
        f"WHERE `partition_date` IS NOT NULL "
        f"ORDER BY `partition_date` DESC"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall() or []
    except Exception as ex:
        print(f"    [WARN] Querying partition_date in `{table}` failed; partition priority disabled: {ex}")
        return None
    dates = [r.get("pd") for r in rows if r.get("pd") is not None]
    return dates


def _collect_from_rows(rows, out: List[str], vid_seen: Set[str], limit: int,
                       exclude_vids: Optional[Set[str]] = None) -> int:
    added = 0
    for r in rows:
        v = r.get("tid")
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        vid = sanitize_vid(s)
        if not vid or vid in vid_seen:
            continue
        if exclude_vids is not None and vid in exclude_vids:
            continue
        vid_seen.add(vid)
        out.append(s)
        added += 1
        if isinstance(limit, int) and limit > 0 and len(out) >= limit:
            break
    return added


def fetch_target_ids(conn, table: str, limit: int,
                     exclude_vids: Optional[Set[str]] = None) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    partition_dates = get_partition_dates(conn, table)

    if not partition_dates:
        base_sql = (
            f"SELECT DISTINCT `target_id` AS tid FROM `{table}` "
            f"WHERE `target_id` IS NOT NULL AND `target_id` <> ''"
        )
        if isinstance(limit, int) and limit > 0:
            sql = base_sql + f" LIMIT {int(limit)}"
        else:
            sql = base_sql
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall() or []
        _collect_from_rows(rows, out, seen, limit, exclude_vids)
        return out
    for pd in partition_dates:
        if isinstance(limit, int) and limit > 0 and len(out) >= limit:
            break
        sql = (
            f"SELECT DISTINCT `target_id` AS tid FROM `{table}` "
            f"WHERE `partition_date` = %s "
            f"AND `target_id` IS NOT NULL AND `target_id` <> ''"
        )
        if isinstance(limit, int) and limit > 0:
            remaining = limit - len(out)
            sql += f" LIMIT {int(remaining)}"
        with conn.cursor() as cur:
            cur.execute(sql, (pd,))
            rows = cur.fetchall() or []
        added = _collect_from_rows(rows, out, seen, limit, exclude_vids)
        print(f"      Partition partition_date={pd}: added {added} target_ids (total {len(out)})")

    return out


def build_target_table(prefix: str, suffix: str) -> str:
    return f"{prefix}{suffix}"


def build_small_subset(
    samples: List[Dict[str, str]],
    small_path: str,
    n_black: int,
    n_white: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    black = [s for s in samples if s.get("tag") == "black"]
    white = [s for s in samples if s.get("tag") == "white"]
    if n_black >= len(black):
        sel_black = list(black)
        if n_black > len(black):
            print(
                f"    [WARN] Only {len(black)} black samples < requested {n_black}; "
                f"using all available samples"
            )
    else:
        sel_black = rng.sample(black, n_black)
    if n_white >= len(white):
        sel_white = list(white)
        if n_white > len(white):
            print(
                f"    [WARN] Only {len(white)} white samples < requested {n_white}; "
                f"using all available samples"
            )
    else:
        sel_white = rng.sample(white, n_white)
    subset = sel_black + sel_white
    with open(small_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    print(
        f"==> Small subset written: {small_path} "
        f"(black={len(sel_black)}, white={len(sel_white)}, total={len(subset)})"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect black/white target_id samples from StarRocks by entity prefix (default 5000/10000)",
    )
    p.add_argument("--black-limit", type=int, default=BLACK_LIMIT_PER_TABLE,
                   help=f"Maximum target_ids per black table (default {BLACK_LIMIT_PER_TABLE})")
    p.add_argument("--white-limit", type=int, default=WHITE_LIMIT_PER_TABLE,
                   help=f"Maximum target_ids per white table (default {WHITE_LIMIT_PER_TABLE})")
    p.add_argument("--out-tag", default="",
                   help="Output tag: <stem>_<tag>_large.json; empty reuses existing names and overwrites files")
    p.add_argument("--prefix", default="",
                   help="Collect only this entity prefix (default all); choices are listed in PREFIX_TO_OUTFILE")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rules = build_suffix_rules(int(args.black_limit), int(args.white_limit))
    prefix_map = {pref: apply_out_tag(fname, args.out_tag)
                  for pref, fname in PREFIX_TO_OUTFILE.items()}
    if args.prefix:
        if args.prefix not in prefix_map:
            print(f"[ERROR] Unknown --prefix {args.prefix!r}; choices: {list(prefix_map)}")
            return 1
        prefix_map = {args.prefix: prefix_map[args.prefix]}
    print("==> Entity prefixes (all->*_large.json, black->*_large_black.json, subset->*_small.json):")
    for pref, fname in prefix_map.items():
        print(f"    - {pref}  ->  data/{fname}  (large + large_black + small)")
    print("==> Suffix rules (longer suffixes first):")
    for suf, tag, lim in rules:
        print(f"    - suffix {suf!r:20s}  tag={tag!r:7s}  maximum {lim} per table")
    print(f"==> VID truncation limit: {VID_SIZE} bytes (UTF-8)")
    print(
        f"==> Small subset sampling: black={N_BLACK_SAMPLE}, white={N_WHITE_SAMPLE}, "
        f"seed={RANDOM_SEED}"
    )
    print("==> Available table/partition rows may yield fewer samples than the configured limits")
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = connect_starrocks()
    target_vids = _load_target_entry_vids()
    if target_vids:
        print(f"==> Keeping IV samples disjoint from mining targets: excluding {len(target_vids)} target vids "
              f"(data/target_cache)")
    else:
        print("==> No target cache found (data/target_cache); target exclusion is skipped during collection "
              "(real_iv panel loading still applies fallback filtering)")
    try:
        for prefix, outfile_name in prefix_map.items():
            print(f"\n========== Processing prefix: {prefix} ==========")
            vid_seen: Set[str] = set()
            samples: List[Dict[str, str]] = []
            raw_total = 0
            truncated_total = 0
            truncation_examples: List[Dict[str, str]] = []
            per_table_stats: List[Dict] = []
            for suffix, tag, limit in rules:
                table = build_target_table(prefix, suffix)
                print(f"\n-- Processing table `{table}` (tag={tag}, limit={limit}) ...")
                if not table_exists(conn, table):
                    print(f"   [SKIP] Table `{table}` does not exist")
                    per_table_stats.append({
                        "table": table, "tag": tag, "exists": False,
                        "raw_count": 0, "added": 0,
                    })
                    continue
                if not has_target_id_column(conn, table):
                    print(f"   [SKIP] Table `{table}` has no target_id column")
                    per_table_stats.append({
                        "table": table, "tag": tag, "exists": True,
                        "has_target_id": False, "raw_count": 0, "added": 0,
                    })
                    continue
                try:
                    exclude = (vid_seen | target_vids) if tag == "black" \
                        else (target_vids or None)
                    raw_ids = fetch_target_ids(conn, table, limit, exclude_vids=exclude)
                except Exception as ex:
                    print(f"   [WARN] target_id extraction failed: {ex}")
                    continue

                added = 0
                tbl_truncated = 0
                for raw in raw_ids:
                    vid = sanitize_vid(raw)
                    if not vid:
                        continue
                    if vid != raw:
                        tbl_truncated += 1
                        truncated_total += 1
                        if len(truncation_examples) < 5:
                            truncation_examples.append({"raw": raw, "vid": vid})
                    if vid in vid_seen:
                        continue
                    vid_seen.add(vid)
                    samples.append({"target_id": vid, "tag": tag})
                    added += 1
                raw_total += len(raw_ids)
                print(
                    f"   Retrieved {len(raw_ids)} raw target_ids; "
                    f"added {added} {tag} samples"
                    + (f" ({tbl_truncated} truncated)" if tbl_truncated else "")
                )
                per_table_stats.append({
                    "table": table, "tag": tag, "exists": True,
                    "has_target_id": True,
                    "raw_count": len(raw_ids),
                    "added": added,
                    "truncated": tbl_truncated,
                })
            stem, ext = os.path.splitext(outfile_name)
            large_path = os.path.join(OUT_DIR, f"{stem}_large{ext}")
            small_path = os.path.join(OUT_DIR, f"{stem}_small{ext}")
            black_path = os.path.join(OUT_DIR, f"{stem}_large_black{ext}")
            with open(large_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            n_black = sum(1 for s in samples if s["tag"] == "black")
            n_white = sum(1 for s in samples if s["tag"] == "white")
            print(
                f"\n==> Prefix {prefix} complete: total={len(samples)} "
                f"(black={n_black}, white={n_white}); raw={raw_total}; "
                f"truncated={truncated_total}"
            )
            print(f"==> Full sample set written: {large_path}")

            black_only = [s for s in samples if s["tag"] == "black"]
            if black_only:
                with open(black_path, "w", encoding="utf-8") as f:
                    json.dump(black_only, f, ensure_ascii=False, indent=2)
                print(f"==> Black sample set written: {black_path} "
                      f"({len(black_only)} entries, same format as full set: "
                      f'[{{"target_id": ..., "tag": "black"}}])')
            else:
                print("==> [WARN] No black samples collected; skipping *_large_black.json")
            build_small_subset(
                samples, small_path,
                n_black=N_BLACK_SAMPLE, n_white=N_WHITE_SAMPLE,
                seed=RANDOM_SEED,
            )
            print("    Per-table statistics:")
            for st in per_table_stats:
                print(f"      - {st}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    print(f"\n[DONE] {time.strftime('%Y-%m-%d %H:%M:%S')} All prefixes processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
