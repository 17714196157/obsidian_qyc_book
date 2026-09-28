# -*- coding: utf-8 -*-
import json
import logging
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
logger = logging.getLogger(__name__)


# ============================================================
# 外部 HTTP API 检索（线上小模型相似样本服务）
# ============================================================

# 外部相似样本 API 地址, 获取后续集
SIM_SAMPLE_API_URL = "http://192.168.0.117:8020/opt2_get_sim_sample"
SIM_SAMPLE_API_TIMEOUT_SEC = 30
def retrieve_from_api(keyword: str) -> List[Tuple[str, str, float]]:
    """调用外部 HTTP API 获取候选 ICD 编码（线上小模型相似样本服务）。

    Args:
        keyword: 手术名称（原始词）

    Returns:
        [(icd_code, description, similarity), ...] 按相似度降序
    """
    payload = json.dumps({"keyword": keyword}).encode("utf-8")
    req = urllib.request.Request(
        SIM_SAMPLE_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=SIM_SAMPLE_API_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, KeyboardInterrupt) as e:
        logger.warning("API 调用失败: %s", e)
        return []

    if data.get("retCode") != 0:
        logger.warning("API 返回错误: retCode=%s", data.get("retCode"))
        return []

    results = data.get("result", [])
    if not isinstance(results, list):
        return []

    # 转换为统一格式: [(icd_code, desc, sim), ...]
    candidates = []
    seen = set()
    for item in results:
        name = item.get("name", "")
        code = item.get("code", "")
        sim = float(item.get("sim", 0.0))
        if code and code not in seen:
            seen.add(code)
            candidates.append((code, name, sim))
    return candidates


# 对照之前小模型的编码结果
SMALL_MODEL_URL = "http://192.168.0.117:8020/opt2"
SMALL_MODEL_TIMEOUT_SEC = 600

def predict_small_model_batch(keywords: List[str]) -> List[dict]:
    """调用小模型编码服务，批量预测 ICD 编码。

    Returns:
        [{"corrName": str, "name": str, "code": str, "score": float}, ...]
    """
    if not keywords:
        return []

    payload = json.dumps({"keyword": keywords}).encode("utf-8")
    req = urllib.request.Request(
        SMALL_MODEL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=SMALL_MODEL_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.warning("小模型 API 调用失败: %s", e)
        return [{"corrName": kw, "name": "", "code": "", "score": 0.0} for kw in keywords]

    op_list = data.get("OperationList", [])
    if not isinstance(op_list, list):
        logger.warning("小模型 API 返回格式异常: %s", str(data)[:200])
        return [{"corrName": kw, "name": "", "code": "", "score": 0.0} for kw in keywords]

    results = []
    for node in op_list:
        corr_name = node.get("corrName", "")
        code_list = node.get("codeList", [])
        if code_list and isinstance(code_list, list) and len(code_list) > 0:
            top = code_list[0]
            results.append({
                "corrName": corr_name,
                "name": top.get("name", ""),
                "code": top.get("code", ""),
                "score": float(top.get("score", 0.0)),
            })
        else:
            results.append({"corrName": corr_name, "name": "", "code": "", "score": 0.0})

    return results


# jev模型http服务
jev_MODEL_URL = "http://192.168.0.181:8002/icd/rerank"
jev_MODEL_TIMEOUT_SEC = 600
def predict_jev_icdrerank(clinical_text, candidates):
    """
    :param clinical_text: 病历文本
    :param candidates: 候选icd编码列表
    :return: icd编码列表
    curl --location --request POST 'http://192.168.0.181:8000/icd/rerank' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "clinical_text": "2型糖尿病伴酮症酸中毒，血糖控制不佳",
        "top_k": 1,
        "alpha": 0.7,
        "candidates": [
            {
                "code": "E11.1",
                "name": "2型糖尿病伴酮症酸中毒",
                "recall_score": 0.83
            },
            {
                "code": "E11.9",
                "name": "2型糖尿病不伴并发症",
                "recall_score": 0.79
            },
            {
                "code": "E10.1",
                "name": "1型糖尿病伴酮症酸中毒",
                "recall_score": 0.72
            },
            {
                "code": "E13.1",
                "name": "其他特指糖尿病伴酮症酸中毒",
                "recall_score": 0.65
            },
            {
                "code": "E11.65",
                "name": "2型糖尿病伴高血糖",
                "recall_score": 0.60
            }
        ]
    }'

    响应麻溜：
    {
        "results": [
            {
                "code": "E11.9",
                "name": "2型糖尿病不伴并发症",
                "orig_index": 1,
                "laya_score": 1.6001,
                "final_score": 1.3678960869565218
            }
        ],
        "latency_ms": 4239.95
    }
    """
    resp = requests.post(jev_MODEL_URL, json={
        "clinical_text": "2型糖尿病伴酮症酸中毒，血糖控制不佳",
        "top_k": 1,
        "alpha": 0.7,
        "candidates": [
            {"code": "E11.1",  "name": "2型糖尿病伴酮症酸中毒", "recall_score": 0.83},
            {"code": "E11.9",  "name": "2型糖尿病不伴并发症",   "recall_score": 0.79},
            {"code": "E10.1",  "name": "1型糖尿病伴酮症酸中毒", "recall_score": 0.72},
            {"code": "E13.1",  "name": "其他特指糖尿病伴酮症酸中毒", "recall_score": 0.65},
            {"code": "E11.65", "name": "2型糖尿病伴高血糖",     "recall_score": 0.60},
        ],
    })
    return {"code":resp.json()["results"][0]["code"], "name":resp.json()["results"][0]["name"]}


################

# -*- coding: utf-8 -*-
"""
对比测试脚本：小模型 ICD 编码 vs RAG召回 + JEV 重排

数据集采用 data_loader.load_all_data_with_split 按时间切分：
  - 训练集（早于 cutoff_date）用于构建检索索引；
  - 验证集（>= cutoff_date）作为测试集，模拟"未来新词"场景；
  - 从而保证检索索引中不存在测试样本本身，避免数据泄漏。
"""
import argparse
import csv
import json
import logging
import random
import time
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

from data_loader import load_all_data_with_split
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1) JEV 调用封装
# ============================================================
def predict_jev_rerank(
    clinical_text: str,
    candidates: List[dict],
    top_k: int = 1,
    alpha: float = 0.7,
) -> dict:
    try:
        resp = requests.post(
            jev_MODEL_URL,
            json={
                "clinical_text": clinical_text,
                "top_k": top_k,
                "alpha": alpha,
                "candidates": candidates,
            },
            timeout=jev_MODEL_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", []) or []
        if not results:
            return {"code": "", "name": "", "raw": data}
        top = results[0]
        return {"code": top.get("code", ""), "name": top.get("name", ""), "raw": data}
    except Exception as e:
        logger.warning("JEV 调用失败: %s", e)
        return {"code": "", "name": "", "raw": None}


# ============================================================
# 2) 测试集构造：从 load_all_data_with_split 的 val_set 转换
# ============================================================
def build_test_set_from_val(
    val_set: List[Dict],
    n: int = 200,
    seed: int = 42,
    dedup: bool = True,
) -> List[Tuple[str, str]]:
    """把 val_set (List[Dict]) 转成 [(query, gt_code), ...]。

    - dedup=True 时对同一原始词去重（保留最新日期的那条），避免同一词被反复评估。
    - n<=0 表示使用全部数据；否则随机采样 n 条。
    """
    if dedup:
        # {原始词: (icd_code, date)}
        best: Dict[str, Tuple[str, int]] = {}
        for r in val_set:
            q = r.get("原始词", "")
            c = r.get("icd_code", "")
            d = r.get("加入日期", 0)
            if not q or not c:
                continue
            if q not in best or d > best[q][1]:
                best[q] = (c, d)
        items = list(best.items())
        items = [(q, c) for q, (c, _) in items]
    else:
        items = [(r["原始词"], r["icd_code"]) for r in val_set
                 if r.get("原始词") and r.get("icd_code")]

    logger.info("验证集(去重后)共 %d 条", len(items))

    if n and n > 0 and n < len(items):
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:n]

    return items


# ============================================================
# 3) 小模型链路评估
# ============================================================
def eval_small_model(test_set: List[Tuple[str, str]], icd_dict: Dict[str, str]) -> dict:
    keywords = [q for q, _ in test_set]
    logger.info("调用小模型接口，共 %d 条 ...", len(keywords))
    t0 = time.time()
    preds = predict_small_model_batch(keywords)
    logger.info("小模型接口耗时 %.1fs", time.time() - t0)

    pred_map = {p.get("corrName", ""): p for p in preds}
    n_correct = 0
    details: List[dict] = []
    for q, gt in test_set:
        p = pred_map.get(q, {})
        pred_code = p.get("code", "")
        pred_name = p.get("name", "") or icd_dict.get(pred_code, "")
        gt_name = icd_dict.get(gt, "")
        ok = (pred_code == gt)
        n_correct += int(ok)
        details.append({
            "query": q,
            "gt": gt,
            "gt_name": gt_name,
            "small_pred": pred_code,
            "small_pred_name": pred_name,
            "small_score": float(p.get("score", 0.0) or 0.0),
            "small_correct": ok,
        })

    return {
        "method": "small_model",
        "total": len(test_set),
        "correct": n_correct,
        "accuracy": n_correct / max(len(test_set), 1),
        "details": details,
    }


# ============================================================
# 4) RAG召回 + JEV 链路评估
# ============================================================
def eval_rag_jev(
    test_set: List[Tuple[str, str]],
    icd_dict: Dict[str, str],
    top_k_recall: int = 10,
    alpha: float = 0.7,
) -> dict:
    n_jev_correct = 0
    n_recall_top1_correct = 0
    n_in_candidates = 0
    details: List[dict] = []

    for q, gt in tqdm(test_set, desc="RAG+JEV"):
        try:
            recalled = retrieve_from_api(q)
        except Exception as e:
            logger.warning("召回失败 [%s]: %s", q, e)
            recalled = []
        recalled = recalled[:top_k_recall]

        item = {
            "query": q,
            "gt": gt,
            "gt_name": icd_dict.get(gt, ""),
            "n_recall": len(recalled),
            "recall_top1": recalled[0][0] if recalled else "",
            "recall_top1_name": recalled[0][1] if recalled else "",
            "in_candidates": False,
            "jev_pred": "",
            "jev_pred_name": "",
        }

        if not recalled:
            details.append(item)
            continue

        codes = [c for c, _, _ in recalled]
        in_candidates = gt in codes
        item["in_candidates"] = in_candidates
        if in_candidates:
            n_in_candidates += 1
        if recalled[0][0] == gt:
            n_recall_top1_correct += 1

        jev_cands = [
            {"code": c, "name": n, "recall_score": float(s or 0.0)}
            for c, n, s in recalled
        ]

        result = predict_jev_rerank(
            clinical_text=q,
            candidates=jev_cands,
            top_k=1,
            alpha=alpha,
        )
        pred = result.get("code", "")
        item["jev_pred"] = pred
        item["jev_pred_name"] = result.get("name", "") or icd_dict.get(pred, "")
        if pred == gt:
            n_jev_correct += 1

        details.append(item)

    return {
        "method": "rag_jev",
        "total": len(test_set),
        "recall_top1_acc": n_recall_top1_correct / max(len(test_set), 1),
        "recall_at_k": n_in_candidates / max(len(test_set), 1),
        "jev_acc": n_jev_correct / max(len(test_set), 1),
        "details": details,
    }


# ============================================================
# 5) 汇总 & 差异分析
# ============================================================
def merge_and_diff(small_res: dict, jev_res: dict) -> dict:
    small_map = {d["query"]: d for d in small_res["details"]}

    merged: List[dict] = []
    for d in jev_res["details"]:
        s = small_map.get(d["query"], {})
        m = {
            "query": d["query"],
            "gt": d["gt"],
            "gt_name": d.get("gt_name", ""),
            "small_pred": s.get("small_pred", ""),
            "small_pred_name": s.get("small_pred_name", ""),
            "small_correct": bool(s.get("small_correct", False)),
            "recall_top1": d["recall_top1"],
            "recall_top1_name": d.get("recall_top1_name", ""),
            "recall_top1_correct": d["recall_top1"] == d["gt"],
            "in_candidates": d["in_candidates"],
            "jev_pred": d["jev_pred"],
            "jev_pred_name": d.get("jev_pred_name", ""),
            "jev_correct": d["jev_pred"] == d["gt"],
        }
        if m["small_correct"] and m["jev_correct"]:
            m["diff_type"] = "两者均对"
        elif not m["small_correct"] and not m["jev_correct"]:
            m["diff_type"] = "两者均错"
        elif m["small_correct"] and not m["jev_correct"]:
            m["diff_type"] = "仅小模型对"
        else:
            m["diff_type"] = "仅JEV对"
        merged.append(m)

    small_only = [m for m in merged if m["diff_type"] == "仅小模型对"]
    jev_only   = [m for m in merged if m["diff_type"] == "仅JEV对"]
    both_ok    = [m for m in merged if m["diff_type"] == "两者均对"]
    both_bad   = [m for m in merged if m["diff_type"] == "两者均错"]

    return {
        "summary": {
            "small_model_acc": small_res["accuracy"],
            "recall_top1_acc": jev_res["recall_top1_acc"],
            "recall_at_k": jev_res["recall_at_k"],
            "rag_jev_acc": jev_res["jev_acc"],
            "both_correct": len(both_ok),
            "both_wrong": len(both_bad),
            "small_only_correct": len(small_only),
            "jev_only_correct": len(jev_only),
        },
        "diff": {
            "small_only_correct": small_only,
            "jev_only_correct": jev_only,
        },
        "details": merged,
    }


# ============================================================
# 6) CSV 导出（含 ICD 名称）
# ============================================================
def save_details_to_csv(details: List[dict], csv_path: str):
    fieldnames = [
        "query", "gt", "gt_name",
        "small_pred", "small_pred_name", "small_correct",
        "recall_top1", "recall_top1_name", "recall_top1_correct", "in_candidates",
        "jev_pred", "jev_pred_name", "jev_correct",
        "diff_type",
    ]
    header_mapping = {
        "query": "原始词",
        "gt": "正确ICD(GT)",
        "gt_name": "正确ICD名称",
        "small_pred": "小模型预测",
        "small_pred_name": "小模型预测名称",
        "small_correct": "小模型是否正确",
        "recall_top1": "召回Top1",
        "recall_top1_name": "召回Top1名称",
        "recall_top1_correct": "召回Top1是否正确",
        "in_candidates": "GT是否在召回集中",
        "jev_pred": "JEV预测",
        "jev_pred_name": "JEV预测名称",
        "jev_correct": "JEV是否正确",
        "diff_type": "结论",
    }
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([header_mapping[k] for k in fieldnames])
            for d in details:
                row = [d.get(k, "") for k in fieldnames]
                for bool_idx in [5, 8, 9, 12]:
                    row[bool_idx] = "是" if row[bool_idx] else "否"
                writer.writerow(row)
        logger.info("逐条对比明细（含 ICD 名称）已保存至 %s", csv_path)
    except Exception as e:
        logger.error("保存 CSV 失败: %s", e)


def print_summary(s: dict, top_n_diff: int = 10) -> None:
    sm = s["summary"]
    print("\n" + "=" * 80)
    print("对比结果汇总")
    print("=" * 80)
    print(f"测试样本数              : {len(s['details'])}")
    print(f"小模型 Top-1 准确率     : {sm['small_model_acc']:.4f}")
    print(f"召回 Top-1 准确率       : {sm['recall_top1_acc']:.4f}")
    print(f"召回@K 覆盖率(理论上限) : {sm['recall_at_k']:.4f}")
    print(f"RAG+JEV Top-1 准确率    : {sm['rag_jev_acc']:.4f}")
    print("-" * 80)
    print(f"两者均正确              : {sm['both_correct']}")
    print(f"两者均错误              : {sm['both_wrong']}")
    print(f"仅小模型正确            : {sm['small_only_correct']}")
    print(f"仅 JEV 正确             : {sm['jev_only_correct']}")
    print("=" * 80)

    def _print_case_list(title: str, cases: List[dict]):
        print("\n" + "-" * 80)
        print(f"{title} (前 {top_n_diff} 条)")
        print("-" * 80)
        if not cases:
            print("无")
        for i, d in enumerate(cases[:top_n_diff]):
            print(f"{i+1}. 原词: {d['query']}")
            print(f"   GT   : [{d['gt']}] {d['gt_name']}")
            print(f"   小模型: [{d['small_pred']}] {d['small_pred_name']}")
            print(f"   JEV  : [{d['jev_pred']}] {d['jev_pred_name']}")

    _print_case_list("【仅JEV正确】案例 (小模型错、JEV对)", s["diff"]["jev_only_correct"])
    _print_case_list("【仅小模型正确】案例 (小模型对、JEV错)", s["diff"]["small_only_correct"])
    print("=" * 80 + "\n")


# ============================================================
# 7) 主入口
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200,
                        help="测试样本数（<=0 表示使用全部验证集）")
    parser.add_argument("--val_cutoff_date", type=int, default=20260701,
                        help="验证集起始日期 YYYYMMDD，>= 该日期的记录为验证集")
    parser.add_argument("--top_k", type=int, default=10, help="召回候选数上限")
    parser.add_argument("--alpha", type=float, default=0.7, help="JEV 融合权重")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_dedup", action="store_true",
                        help="不对验证集做原始词去重（默认去重，取同一原始词最新记录）")
    parser.add_argument("--out_json", type=str, default="compare_result.json")
    parser.add_argument("--out_csv", type=str, default="compare_details.csv")
    args = parser.parse_args()

    # --- 按时间切分加载数据 ---
    logger.info("按时间切分加载数据 (cutoff=%d) ...", args.val_cutoff_date)
    icd_dict, val_set, raw_term_index, icd_desc_index = load_all_data_with_split(
        val_cutoff_date=args.val_cutoff_date,
    )
    logger.info(
        "icd_dict=%d, val_set=%d, raw_term_index=%d, icd_desc_index=%d",
        len(icd_dict), len(val_set), len(raw_term_index), len(icd_desc_index),
    )

    # --- 构造测试集 ---
    test_set = build_test_set_from_val(
        val_set, n=args.n, seed=args.seed, dedup=(not args.no_dedup),
    )
    if not test_set:
        raise RuntimeError("测试集为空，请检查 val_cutoff_date 设置")

    # --- 小模型 ---
    small_res = eval_small_model(test_set, icd_dict)
    logger.info(
        "小模型 Top-1: %.4f (%d/%d)",
        small_res["accuracy"], small_res["correct"], small_res["total"],
    )

    # --- RAG + JEV ---
    jev_res = eval_rag_jev(test_set, icd_dict,
                           top_k_recall=args.top_k, alpha=args.alpha)
    logger.info("召回 Top-1 准确率: %.4f", jev_res["recall_top1_acc"])
    logger.info("召回@%d 覆盖率   : %.4f", args.top_k, jev_res["recall_at_k"])
    logger.info("RAG+JEV Top-1    : %.4f", jev_res["jev_acc"])

    # --- 汇总 ---
    report = {
        "config": vars(args),
        "data_split": {
            "val_cutoff_date": args.val_cutoff_date,
            "val_set_raw": len(val_set),
            "test_set_size": len(test_set),
        },
        **merge_and_diff(small_res, jev_res),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("完整 JSON 结果已写入 %s", args.out_json)

    save_details_to_csv(report["details"], args.out_csv)
    print_summary(report, top_n_diff=10)


if __name__ == "__main__":
    main()