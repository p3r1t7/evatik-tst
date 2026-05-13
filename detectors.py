# detectors.py -------------------------------------
# ... ----------- v3.5
"""
Все статистические детекторы аномалий (Версия 3.5)
"""
import pandas as pd
import numpy as np
import math
from collections import Counter

# ============================================================
# 1. ПОРОГОВЫЙ МЕТОД (Threshold)
# ============================================================
class ThresholdDetector:
    def __init__(self, percentile=99):
        self.percentile = percentile
        self.threshold = None
    
    def fit(self, baseline_df):
        filtered = baseline_df.copy()
        filtered['timestamp'] = pd.to_datetime(filtered['timestamp'])
        filtered.set_index('timestamp', inplace=True)
        counts = filtered.resample('1min').size()
        self.threshold = np.percentile(counts, self.percentile)
        print(f"    Порог ({self.percentile}%): {self.threshold:.1f} событий/мин")
        return self
    
    def detect(self, test_df):
        test_df = test_df.copy()
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df.set_index('timestamp', inplace=True)
        counts = test_df.resample('1min').size().reset_index()
        counts.columns = ['timestamp', 'count']
        counts['is_anomaly'] = counts['count'] > self.threshold
        counts['score'] = counts['count'] / max(self.threshold, 1)
        return counts

# ============================================================
# 2. СКОЛЬЗЯЩАЯ СРЕДНЯЯ (ИСПРАВЛЕНО: обработка пустого baseline)
# ============================================================
class MovingAverageDetector:
    def __init__(self, windows=[5, 10, 15], percentile_threshold=95):
        self.windows = windows
        self.percentile_threshold = percentile_threshold
        self.baseline_max = None
    
    def fit(self, baseline_df):
        baseline_df = baseline_df.copy()
        baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp'])
        baseline_df.set_index('timestamp', inplace=True)
        counts = baseline_df.resample('1min').size()
        
        self.baseline_max = np.percentile(counts, self.percentile_threshold)
        print(f"    Базовый уровень (P{self.percentile_threshold}): {self.baseline_max:.1f} событий/мин")
        return self
    
    def detect(self, test_df):
        test_df = test_df.copy()
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df.set_index('timestamp', inplace=True)
        counts = test_df.resample('1min').size().reset_index()
        counts.columns = ['timestamp', 'count']
        
        # ИСПРАВЛЕНИЕ v3.5: Если baseline пустой (P95=0), то любое событие считаем аномалией
        if self.baseline_max == 0:
            effective_baseline = 0
            print("    ⚠️ Baseline пуст (P95=0). Любое событие считается аномалией.")
        else:
            effective_baseline = self.baseline_max
        
        all_alerts = []
        for window in self.windows:
            counts_copy = counts.copy()
            counts_copy[f'ma_{window}'] = counts_copy['count'].rolling(window, min_periods=1).mean()
            
            # Логика: Если baseline пустой -> count > 0. Если нет -> count > baseline и count > MA
            if effective_baseline == 0:
                counts_copy['is_anomaly'] = counts_copy['count'] > 0
            else:
                counts_copy['is_anomaly'] = (counts_copy['count'] > counts_copy[f'ma_{window}'] * 1.5) & \
                                            (counts_copy['count'] > effective_baseline)
            
            alerts = counts_copy[counts_copy['is_anomaly']].copy()
            if not alerts.empty:
                alerts['window'] = window
                alerts['score'] = alerts['count'] / max(effective_baseline, 1)
                all_alerts.append(alerts[['timestamp', 'count', 'window', 'is_anomaly', 'score']])
        
        if all_alerts:
            result = pd.concat(all_alerts)
            result = result.sort_values('score', ascending=False)
            return result.drop_duplicates('timestamp', keep='first')
        return pd.DataFrame()

# ============================================================
# 3. Z-SCORE
# ============================================================
class ZScoreDetector:
    def __init__(self, z_threshold=2.5):
        self.z_threshold = z_threshold
        self.mean = None
        self.std = None
    
    def fit(self, baseline_df):
        baseline_df = baseline_df.copy()
        baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp'])
        baseline_df.set_index('timestamp', inplace=True)
        counts = baseline_df.resample('1min').size()
        self.mean = counts.mean()
        self.std = counts.std()
        if self.std == 0: self.std = 1
        print(f"    Среднее: {self.mean:.2f}, Std: {self.std:.2f}")
        return self
    
    def detect(self, test_df):
        test_df = test_df.copy()
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df.set_index('timestamp', inplace=True)
        counts = test_df.resample('1min').size().reset_index()
        counts.columns = ['timestamp', 'count']
        counts['zscore'] = (counts['count'] - self.mean) / self.std
        counts['is_anomaly'] = abs(counts['zscore']) > self.z_threshold
        counts['score'] = abs(counts['zscore'])
        return counts

# ============================================================
# 4. IQR
# ============================================================
class IQRDetector:
    def __init__(self, multipliers=[1.5, 2.0]):
        self.multipliers = multipliers
        self.q1 = None
        self.q3 = None
        self.iqr = None
    
    def fit(self, baseline_df):
        baseline_df = baseline_df.copy()
        baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp'])
        baseline_df.set_index('timestamp', inplace=True)
        counts = baseline_df.resample('1min').size()
        self.q1 = counts.quantile(0.25)
        self.q3 = counts.quantile(0.75)
        self.iqr = self.q3 - self.q1
        return self
    
    def detect(self, test_df):
        test_df = test_df.copy()
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df.set_index('timestamp', inplace=True)
        counts = test_df.resample('1min').size().reset_index()
        counts.columns = ['timestamp', 'count']
        
        all_alerts = []
        for mult in self.multipliers:
            upper = self.q3 + mult * self.iqr
            alerts = counts[counts['count'] > upper].copy()
            if not alerts.empty:
                alerts['multiplier'] = mult
                alerts['is_anomaly'] = True
                alerts['score'] = alerts['count'] / max(self.q3, 1)
                all_alerts.append(alerts[['timestamp', 'count', 'multiplier', 'is_anomaly', 'score']])
        
        if all_alerts:
            result = pd.concat(all_alerts)
            result = result.sort_values('score', ascending=False)
            return result.drop_duplicates('timestamp', keep='first')
        return pd.DataFrame()

# ============================================================
# 5. ЭНТРОПИЯ (ИСПРАВЛЕНО: исключен message)
# ============================================================
class EntropyDetector:
    def __init__(self, entropy_threshold=4.5):
        self.entropy_threshold = entropy_threshold
        self.text_col = None
        self.has_baseline_text = False
    
    @staticmethod
    def _shannon_entropy(text):
        if not text or len(str(text)) == 0: return 0
        text = str(text)
        counter = Counter(text)
        length = len(text)
        entropy = 0
        for count in counter.values():
            prob = count / length
            entropy -= prob * math.log2(prob)
        return entropy
    
    def fit(self, baseline_df):
        # ИСПРАВЛЕНИЕ v3.5: Убрано 'message', так как оно дает слишком много шума (FP)
        possible_cols = ['scriptblock', 'command_line', 'image', 'data_0']
        
        for col in possible_cols:
            if col in baseline_df.columns:
                if baseline_df[col].dropna().shape[0] > 0:
                    self.text_col = col
                    self.has_baseline_text = True
                    print(f"    Используем колонку для энтропии: {col}")
                    break
        
        if not self.has_baseline_text:
            print("    ⚠️ В baseline нет специфичных текстовых данных (ScriptBlock/CommandLine).")
        
        return self
    
    def detect(self, test_df):
        active_col = self.text_col
        
        # Если в baseline не нашли текст, ищем в test_df (только специфичные колонки)
        if not self.has_baseline_text:
            possible_cols = ['scriptblock', 'command_line', 'image']
            for col in possible_cols:
                if col in test_df.columns and test_df[col].dropna().shape[0] > 0:
                    active_col = col
                    print(f"    Нашли текст в атаке: {col}")
                    break
        
        if active_col is None or active_col not in test_df.columns:
            print("    ⚠️ Нет подходящих текстовых данных для анализа энтропии.")
            return pd.DataFrame()
        
        test_df = test_df.copy()
        test_df['entropy'] = test_df[active_col].fillna('').apply(self._shannon_entropy)
        test_df['is_anomaly'] = test_df['entropy'] > self.entropy_threshold
        
        anomalies = test_df[test_df['is_anomaly']].copy()
        if not anomalies.empty:
            anomalies['score'] = anomalies['entropy'] / max(self.entropy_threshold, 1)
            return anomalies[['timestamp', 'entropy', 'is_anomaly', 'score']].copy()
        
        return pd.DataFrame()

# ============================================================
# 6. РЕДКИЕ СОБЫТИЯ (ИСПРАВЛЕНО: сравнение частоты)
# ============================================================
class RareEventDetector:
    CRITICAL_EVENTS = {4720, 4732, 4672, 4698, 4700, 7045, 4697, 4104, 4688, 4624}
    
    def __init__(self):
        self.baseline_ids = None
        self.baseline_freq = None # Частота событий в минуту
    
    def fit(self, baseline_df):
        if 'event_id' in baseline_df.columns:
            self.baseline_ids = set(baseline_df['event_id'].unique())
            
            # Считаем длительность baseline в минутах
            if 'timestamp' in baseline_df.columns:
                baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp'])
                duration = (baseline_df['timestamp'].max() - baseline_df['timestamp'].min()).total_seconds() / 60
                if duration == 0: duration = 1
                
                # Считаем частоту для каждого ID
                counts = baseline_df['event_id'].value_counts()
                self.baseline_freq = counts / duration
            else:
                self.baseline_freq = baseline_df['event_id'].value_counts()
                
        return self
    
    def detect(self, test_df):
        if 'event_id' not in test_df.columns or self.baseline_freq is None:
            return pd.DataFrame()
        
        critical_test = test_df[test_df['event_id'].isin(self.CRITICAL_EVENTS)].copy()
        if critical_test.empty: return pd.DataFrame()
        
        # Считаем длительность теста
        if 'timestamp' in test_df.columns:
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            duration_test = (test_df['timestamp'].max() - test_df['timestamp'].min()).total_seconds() / 60
            if duration_test == 0: duration_test = 1
        else:
            duration_test = 1
            
        rare_events = []
        test_counts = critical_test['event_id'].value_counts()
        
        for eid, count in test_counts.items():
            test_freq = count / duration_test
            base_freq = self.baseline_freq.get(eid, 0)
            
            # Если в baseline события не было вообще
            if base_freq == 0:
                # Добавляем все события этого типа
                subset = critical_test[critical_test['event_id'] == eid]
                for _, row in subset.iterrows():
                    row['is_anomaly'] = True
                    row['score'] = 1.0
                    rare_events.append(row)
            else:
                # Если частота выросла в 3+ раза
                if test_freq > base_freq * 3:
                    subset = critical_test[critical_test['event_id'] == eid]
                    for _, row in subset.iterrows():
                        row['is_anomaly'] = True
                        row['score'] = test_freq / base_freq
                        rare_events.append(row)
        
        if rare_events:
            result = pd.DataFrame(rare_events)
            return result[['timestamp', 'event_id', 'is_anomaly', 'score']]
        return pd.DataFrame()

# ============================================================
# 7. КОРРЕЛЯЦИЯ (ДОБАВЛЕНА ЦЕПОЧКА WINRM)
# ============================================================
class CorrelationDetector:
    def __init__(self, time_window_sec=300):
        self.time_window = time_window_sec
    
    def fit(self, baseline_df):
        return self
    
    def detect(self, test_df):
        if 'event_id' not in test_df.columns or 'timestamp' not in test_df.columns:
            return pd.DataFrame()
        
        test_df = test_df.copy()
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df = test_df.sort_values('timestamp')
        
        chains = []
        
        # 1. WINRM / Lateral Movement: Logon (4624) -> Process (4688)
        logons = test_df[test_df['event_id'] == 4624]
        for _, logon in logons.iterrows():
            logon_time = logon['timestamp']
            proc = test_df[
                (test_df['event_id'] == 4688) & 
                (test_df['timestamp'] > logon_time) & 
                (test_df['timestamp'] <= logon_time + pd.Timedelta(seconds=self.time_window))
            ]
            if len(proc) > 0:
                chains.append({'timestamp': logon_time, 'event_id': '4624→4688', 'is_anomaly': True, 'score': 1.0})

        # 2. Scheduled Task (4698 -> 4700)
        tasks = test_df[test_df['event_id'] == 4698]
        for _, task in tasks.iterrows():
            task_time = task['timestamp']
            enabled = test_df[
                (test_df['event_id'] == 4700) & 
                (test_df['timestamp'] > task_time) & 
                (test_df['timestamp'] <= task_time + pd.Timedelta(seconds=self.time_window))
            ]
            if len(enabled) > 0:
                chains.append({'timestamp': task_time, 'event_id': '4698→4700', 'is_anomaly': True, 'score': 1.0})

        # 3. PowerShell Script -> Process (4104 -> 4688)
        scripts = test_df[test_df['event_id'] == 4104]
        for _, script in scripts.iterrows():
            script_time = script['timestamp']
            proc = test_df[
                (test_df['event_id'] == 4688) & 
                (test_df['timestamp'] > script_time) & 
                (test_df['timestamp'] <= script_time + pd.Timedelta(seconds=self.time_window))
            ]
            if len(proc) > 0:
                chains.append({'timestamp': script_time, 'event_id': '4104→4688', 'is_anomaly': True, 'score': 1.0})

        # 4. User Creation (4720 -> 4732)
        creations = test_df[test_df['event_id'] == 4720]
        for _, create in creations.iterrows():
            create_time = create['timestamp']
            admin = test_df[
                (test_df['event_id'] == 4732) & 
                (test_df['timestamp'] > create_time) & 
                (test_df['timestamp'] <= create_time + pd.Timedelta(seconds=self.time_window))
            ]
            if len(admin) > 0:
                chains.append({'timestamp': create_time, 'event_id': '4720→4732', 'is_anomaly': True, 'score': 1.0})
        
        if chains:
            return pd.DataFrame(chains)
        return pd.DataFrame()