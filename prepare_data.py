import json
import os

def get_transcript(files_dir: list, script_key: str): 
    transcripts = []
    for file in files_dir: 
        try: 
            # Đọc từng file JSON
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Duyệt qua các mục trong file JSON
                for key, value in data.items():
                    script = value.get(script_key, "")  # Lấy script (nếu có)
                    if script:
                        transcripts.append(script)  # Thêm vào danh sách

        except Exception as e:
            print(f"Không thể đọc file {file}: {e}")
    
    return transcripts

import os

def create_vocab_file(text_list, file_path: str):
    vocab = []
    
    # Tạo thư mục nếu chưa tồn tại
    folder = os.path.dirname(file_path)
    os.makedirs(folder, exist_ok=True)

    # Mở file và ghi vocab
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("<pad>\n")
        file.write("<sos>\n")
        file.write("<eos>\n")

        for text in text_list:
            words = text.split(" ")
            for word in words:
                if word not in vocab:
                    vocab.append(word)
                    file.write(word + "\n")

        file.write("<unk>\n")
        file.write("<blank>\n")
    
    print(f"Vocab đã được lưu tại: {file_path}")
    return file_path

        

def create_word2index_file(saved_folder, file_vocab_path: str):
    with open(file_vocab_path, "r", encoding="utf-8") as file:
        words = [line.strip() for line in file]

    word2index = dict()
    index2word = dict()

    for index, word in enumerate(words):
        word2index[word] = index
        index2word[index] = word
    
    word2index_dir = os.path.join(saved_folder, "word2index.json")
    index2word_dir = os.path.join(saved_folder, "index2word.json")
    with open(word2index_dir, "w", encoding="utf-8") as file:
        json.dump(word2index, file, ensure_ascii=False, indent=4)
    
    print(f"File word2index đã được lưu tại: {word2index_dir}")

    with open(index2word_dir, "w", encoding="utf-8") as file:
        json.dump(index2word, file, ensure_ascii=False, indent=4)

    print(f"File index2word đã được lưu tại: {index2word_dir}")

    return [word2index_dir,index2word_dir]