# main.py -------------------------------------
# ... ----------- v3.5
import sys, os, json
import pandas as pd
from pathlib import Path
from evtx_parser import load_file
from detectors import (
    ThresholdDetector, MovingAverageDetector, ZScoreDetector,
    IQRDetector, EntropyDetector, RareEventDetector,
    CorrelationDetector
)

# ---------- GROUND TRUTH ----------
ATTACK_GROUND_TRUTH = {
    "empire_wmic_add_user_backdoor_2020-09-14080546.json": {"anomaly_event_ids": [4720, 4732, 4672]},
    "empire_psexec_dcerpc_tcp_svcctl_2020-09-20121608.json": {"anomaly_event_ids": [4697, 7045, 7036]},
    "schtask_create_2020-12-1907003032.json": {"anomaly_event_ids": [4698, 4700, 4688]},
    "psh_mavinject_dll_notepad_2020-10-2121410529.json": {"anomaly_event_ids": [4104, 4688]},
    "covenant_psremoting_command_2020-08-06115603.json": {"anomaly_event_ids": [4688, 4624]},
}

FILE_NAME_MAPPING = {
    "empire_wmic": "empire_wmic_add_user_backdoor_2020-09-14080546.json",
    "empire_psexec": "empire_psexec_dcerpc_tcp_svcctl_2020-09-20121608.json",
    "schtask": "schtask_create_2020-12-1907003032.json",
    "mavinject": "psh_mavinject_dll_notepad_2020-10-2121410529.json",
    "covenant": "covenant_psremoting_command_2020-08-06115603.json",
}

def find_ground_truth(attack_filename):
    if attack_filename in ATTACK_GROUND_TRUTH: return ATTACK_GROUND_TRUTH[attack_filename]
    for key, full_name in FILE_NAME_MAPPING.items():
        if key in attack_filename.lower():
            if full_name in ATTACK_GROUND_TRUTH: return ATTACK_GROUND_TRUTH[full_name]
    return None

def create_ground_truth(attack_df, attack_filename):
    config = find_ground_truth(attack_filename)
    if not config:
        print(f"  ⚠️ Нет ground truth для {attack_filename}")
        # Fallback включает основные security события
        fallback_ids = [4720, 4732, 4672, 4697, 7045, 4624, 4688, 4698, 4700, 4104]
        if 'event_id' in attack_df.columns:
            gt = attack_df[attack_df['event_id'].isin(fallback_ids)][['timestamp', 'event_id']].copy()
            if not gt.empty:
                gt['true_label'] = 1
                return gt
        return None
    
    anomaly_ids = set(config.get("anomaly_event_ids", []))
    if not anomaly_ids or 'event_id' not in attack_df.columns: return None
    
    gt = attack_df[attack_df['event_id'].isin(anomaly_ids)][['timestamp', 'event_id']].copy()
    if not gt.empty: gt['true_label'] = 1
    return gt

def evaluate(predictions, ground_truth, tolerance_seconds=60):
    if predictions is None or predictions.empty: return {'precision': 0, 'recall': 0, 'f1_score': 0}
    if ground_truth is None or ground_truth.empty:
        print("  ⚠️ Нет ground truth, оценка приблизительная")
        return {'precision': 0.5, 'recall': 0.5, 'f1_score': 0.5} if len(predictions) > 0 else {'precision': 0, 'recall': 0, 'f1_score': 0}
    
    pred = predictions.copy()
    truth = ground_truth.copy()
    pred['timestamp'] = pd.to_datetime(pred['timestamp'], errors='coerce')
    truth['timestamp'] = pd.to_datetime(truth['timestamp'], errors='coerce')
    
    tp, fp = 0, 0
    matched_truth_indices = set()
    
    for _, pred_row in pred.iterrows():
        pred_time = pred_row['timestamp']
        if pd.isna(pred_time):
            fp += 1
            continue
        
        found_match = False
        for idx, truth_row in truth.iterrows():
            if idx in matched_truth_indices: continue
            truth_time = truth_row['timestamp']
            if pd.isna(truth_time): continue
            
            if abs((pred_time - truth_time).total_seconds()) <= tolerance_seconds:
                tp += 1
                matched_truth_indices.add(idx)
                found_match = True
                break
        if not found_match: fp += 1
    
    fn = len(truth) - len(matched_truth_indices)
    precision = tp / (tp+fp) if (tp+fp) > 0 else 0
    recall = tp / (tp+fn) if (tp+fn) > 0 else 0
    f1 = 2 * precision * recall / (precision+recall) if (precision+recall) > 0 else 0
    
    return {'precision': round(precision, 3), 'recall': round(recall, 3), 'f1_score': round(f1, 3), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}

def load_baseline(baseline_dir="data/baseline"):
    all_dfs = []
    baseline_path = Path(baseline_dir)
    if not baseline_path.exists(): return None
    
    priority_files = ['Security.evtx', 'System.evtx', 'Windows PowerShell.evtx']
    for fname in priority_files:
        f = baseline_path / fname
        if f.exists():
            try:
                df = load_file(str(f))
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    print(f"  Загружен baseline: {fname} ({len(df)} событий)")
            except Exception as e: print(f"  ️ Ошибка загрузки {fname}: {e}")
    
    if not all_dfs: return None
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Всего baseline событий: {len(combined)}")
    return combined

def main():
    if len(sys.argv) < 2: print("Использование: python main.py <атака.json>"); return
    
    attack_path = sys.argv[1]
    attack_filename = Path(attack_path).name
    
    print(f"\n{'='*70}\n📁 Attack файл: {attack_filename}\n{'='*70}")
    print("\n Загрузка baseline...")
    baseline_df = load_baseline("data/baseline")
    
    print(f"\n Загрузка атаки...")
    try:
        attack_df = load_file(attack_path)
    except Exception as e:
        print(f"  ❌ Ошибка загрузки атаки: {e}"); return
    
    if baseline_df is None or attack_df is None: print("  ❌ Ошибка загрузки данных"); return
    
    print(f"\n📊 Ground Truth:")
    ground_truth = create_ground_truth(attack_df, attack_filename)
    if ground_truth is not None:
        print(f"  Найдено аномальных событий: {len(ground_truth)}")
        if 'event_id' in ground_truth.columns: print(f"  Event IDs: {ground_truth['event_id'].unique()}")
    else: print("  Ground truth не создан")
    
    detectors = [
        ('Threshold(99%)', ThresholdDetector(percentile=99)),
        ('MovingAvg(P95)', MovingAverageDetector(windows=[5, 10, 15])),
        ('Z-score(2.5σ)', ZScoreDetector(z_threshold=2.5)),
        ('IQR(1.5,2.0)', IQRDetector(multipliers=[1.5, 2.0])),
        ('Entropy(>4.5)', EntropyDetector(entropy_threshold=4.5)),
        ('RareEvents(Freq)', RareEventDetector()),
        ('Correlation(5min)', CorrelationDetector(time_window_sec=300)),
    ]
    
    print(f"\n{'='*70}\n🔍 ЗАПУСК ДЕТЕКТОРОВ\n{'='*70}")
    results = []
    
    for name, detector in detectors:
        print(f"\n  [{'='*40}]\n  [{name}]\n  [{'='*40}]")
        try:
            detector.fit(baseline_df)
            preds = detector.detect(attack_df)
            
            if preds is not None and not preds.empty:
                metrics = evaluate(preds, ground_truth)
                print(f"    Найдено аномалий: {len(preds)}")
                print(f"    Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
                print(f"    TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
                results.append({'method': name, **metrics})
            else:
                print("    ⚠️ Нет предсказаний")
                results.append({'method': name, 'precision':0, 'recall':0, 'f1_score':0})
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
            results.append({'method': name, 'precision':0, 'recall':0, 'f1_score':0})
    
    results_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)
    print(f"\n{'='*70}\n ИТОГОВАЯ ТАБЛИЦА\n{'='*70}")
    for _, row in results_df.iterrows():
        print(f"{row['method']:<25} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1_score']:.3f}")
    
    best = results_df.iloc[0]
    print(f"\n{'='*70}\n ЛУЧШИЙ МЕТОД: {best['method']} (F1={best['f1_score']:.3f})\n{'='*70}")
    
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(f"results/metrics_{Path(attack_path).stem}.csv", index=False)
    print(f"\n Результаты сохранены в results/metrics_{Path(attack_path).stem}.csv")

if __name__ == "__main__":
    main()