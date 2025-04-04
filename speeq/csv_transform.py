import json
import csv
import os

voice_folder_path = r"C:\paper\raw_data\Vietnamese-Speech-to-Text-datasets\Common-Voice\voices"
json_folder_path  = r"C:\paper\raw_data\Vietnamese-Speech-to-Text-datasets\Common-Voice"
csv_folder_path   = r"C:\paper\raw_data\Vietnamese-Speech-to-Text-datasets\Common-Voice\newdata"

types = ['train', 'dev', 'test']

for t in types:
    json_file = os.path.join(json_folder_path, f"{t}.json")   # vd: train.json, dev.json, test.json
    csv_file  = os.path.join(csv_folder_path,  f"{t}.csv")    # vd: train.csv, dev.csv, test.csv
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'text'])  # Header
        
        for key, value in data.items():
            # Lấy đường dẫn đến file âm thanh
            file_path = os.path.join(voice_folder_path, value.get('voice', ''))
            # Lấy text (script)
            text = value.get('script', '')
            # Ghi vào CSV
            writer.writerow([file_path, text])
    
    print(f"Done writing {csv_file}")
