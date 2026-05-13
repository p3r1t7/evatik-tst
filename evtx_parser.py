"""
Парсер .evtx и .json файлов
"""
import pandas as pd
import json
import re
from pathlib import Path

def parse_evtx(filepath):
    """Парсинг .evtx файла"""
    try:
        from Evtx.Evtx import Evtx
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("Установите: pip install python-evtx")
    
    print(f"  Чтение: {Path(filepath).name}")
    
    events = []
    namespaces = {'ns': 'http://schemas.microsoft.com/win/2004/08/events/event'}
    
    with Evtx(filepath) as log:
        for record in log.records():
            try:
                xml_str = record.xml()
                root = ET.fromstring(xml_str)
                event = {}
                
                system = root.find('.//ns:System', namespaces)
                if system is None:
                    system = root.find('.//System')
                
                if system is not None:
                    eid = system.find('.//ns:EventID', namespaces)
                    if eid is None:
                        eid = system.find('.//EventID')
                    if eid is not None:
                        event['event_id'] = int(eid.text)
                    
                    tc = system.find('.//ns:TimeCreated', namespaces)
                    if tc is None:
                        tc = system.find('.//TimeCreated')
                    if tc is not None:
                        event['timestamp'] = tc.get('SystemTime')
                    
                    comp = system.find('.//ns:Computer', namespaces)
                    if comp is None:
                        comp = system.find('.//Computer')
                    if comp is not None and comp.text:
                        event['computer'] = comp.text
                
                ed = root.find('.//ns:EventData', namespaces)
                if ed is None:
                    ed = root.find('.//EventData')
                
                if ed is not None:
                    data_list = []
                    for data in ed.findall('.//Data'):
                        if data.text:
                            data_list.append(data.text.strip())
                    
                    for i, val in enumerate(data_list[:3]):
                        event[f'data_{i}'] = val
                    
                    # MSSQL (18456)
                    if event.get('event_id') == 18456 and data_list:
                        user = data_list[0]
                        user = user.replace('&lt;string&gt;', '').replace('&lt;/string&gt;', '')
                        event['username'] = user
                        
                        if len(data_list) >= 3:
                            ip_match = re.search(r'\[CLIENT:\s*([0-9.]+)\]', data_list[2])
                            if ip_match:
                                event['ip_address'] = ip_match.group(1)
                    
                    # PowerShell (4104)
                    if event.get('event_id') == 4104:
                        sb = root.find('.//*[@Name="ScriptBlockText"]')
                        if sb is not None and sb.text:
                            event['scriptblock'] = sb.text
                
                if event:
                    events.append(event)
            except Exception:
                pass
    
    if not events:
        raise ValueError(f"Не удалось извлечь события из {filepath}")
    
    df = pd.DataFrame(events)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"  Извлечено событий: {len(df)}")
    return df


def parse_json(filepath):
    """Парсинг JSON Lines файлов (OTRF Security Datasets)"""
    print(f"  Чтение JSON: {Path(filepath).name}")
    
    events = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                
                event = {
                    'timestamp': raw.get('@timestamp') or raw.get('TimeCreated'),
                    'event_id': raw.get('EventID'),
                    'computer': raw.get('Hostname') or raw.get('computer_name'),
                    'log_name': raw.get('Channel') or raw.get('log_name'),
                    'source_name': raw.get('SourceName') or raw.get('source_name'),
                    'message': raw.get('Message', '')
                }
                
                # Извлекаем event_data
                event_data = raw.get('event_data', {})
                if not event_data:
                    for key in ['Image', 'ProcessId', 'TargetImage', 'SourceImage', 
                               'GrantedAccess', 'UtcTime', 'TargetObject']:
                        if key in raw:
                            event_data[key] = raw[key]
                
                for key, value in event_data.items():
                    event[key.lower()] = value
                
                events.append(event)
                
            except json.JSONDecodeError:
                continue
    
    if not events:
        raise ValueError(f"Не удалось извлечь события из {filepath}")
    
    df = pd.DataFrame(events)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"  Извлечено событий: {len(df)}")
    return df


def load_file(filepath):
    """Автоматически определяет формат и загружает файл"""
    ext = Path(filepath).suffix.lower()
    
    if ext == '.evtx':
        return parse_evtx(filepath)
    elif ext == '.json':
        return parse_json(filepath)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")