import json
import os
import argparse
import random
from dataclasses import dataclass

from speechbrain.dataio.dataio import (
    read_audio_info
)
from speechbrain.utils.logger import get_logger


logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, required=True)
parser.add_argument('--saved_folder', type=str, required=True)
parser.add_argument('--create_vocab', type=lambda x: x.lower() == 'true', required=True)
parser.add_argument('--skip_prep', type=lambda x: x.lower() == 'true', required=True)


def prepare_vinbigdata_vlsp(
    data_folder,
    saved_folder,
    create_vocab=False,
    skip_prep=False
):
    if skip_prep:
        return
    data_folder = data_folder
    saved_folder = saved_folder

    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)


    logger.info("Data_preparation...")
    
    file_names = list(set([file_name[:-4] for file_name in os.listdir(data_folder)]))
    random.shuffle(file_names)
    train_valid_index = int(0.8 * len(file_names))
    train_valid_files = file_names[:train_valid_index]
    test_files = file_names[train_valid_index:]
    train_index = int(0.8 * len(train_valid_files))
    train_files = train_valid_files[:train_index]
    valid_files = train_valid_files[train_index:]

    types = ["train", "valid", "test"]
    text_list = []
    for type in types:
        file_path = os.path.join(saved_folder, type + ".json")
        if os.path.exists(file_path):
            logger.info("Csv file %s already exists, not recreating." % file_path)
            return

        msg = "Creating file %s in %s..." % (type, file_path)
        logger.info(msg)
        if type == "train":
            train = {}
            for ith, file_name in enumerate(train_files):
                wav_file = os.path.join(data_folder, file_name + ".wav")
                transcript = open(os.path.join(data_folder, file_name + ".txt"), "r", encoding="utf-8").read()
                text_list.append(transcript)
                info = process_line(wav_file, transcript)
                train[ith] = {
                    "duration": info.duration,
                    "wav_file": info.file_path,
                    "transcript": info.words
                }

            with open(os.path.join(file_path), "w", encoding="utf-8") as file:
                json.dump(train, file, ensure_ascii=False, indent=4)

        elif type == "valid":
            valid = {}
            for ith, file_name in enumerate(valid_files):
                wav_file = os.path.join(data_folder, file_name + ".wav")
                transcript = open(os.path.join(data_folder, file_name + ".txt"), "r", encoding="utf-8").read()
                text_list.append(transcript)
                info = process_line(wav_file, transcript)
                valid[ith] = {
                    "duration": info.duration,
                    "wav_file": info.file_path,
                    "transcript": info.words
                }

            with open(os.path.join(file_path), "w", encoding="utf-8") as file:
                json.dump(valid, file, ensure_ascii=False, indent=4)

        else:
            test = {}
            for ith, file_name in enumerate(test_files):
                wav_file = os.path.join(data_folder, file_name + ".wav")
                transcript = open(os.path.join(data_folder, file_name + ".txt"), "r", encoding="utf-8").read()
                text_list.append(transcript)
                info = process_line(wav_file, transcript)
                test[ith] = {
                    "duration": info.duration,
                    "wav_file": info.file_path,
                    "transcript": info.words
                }

            with open(os.path.join(file_path), "w", encoding="utf-8") as file:
                json.dump(test, file, ensure_ascii=False, indent=4)

        msg = "%s successfully created!" % (file_path)
        logger.info(msg)

    if create_vocab:
        logger.info("Creating vocab...")
        create_vocab_file(text_list, saved_folder)
        create_word2index_file(saved_folder)
        logger.info("Vocab sucessfully created!")


def create_vocab_file(text_list, saved_folder):
    vocab = []
    with open(os.path.join(saved_folder, "vocab.txt"), "w", encoding="utf-8") as file:
        file.write("<pad>\n")
        file.write("<s>\n")
        file.write("</s>\n")
        for text in text_list:
            words = text.split(" ")
            for word in words:
                if word not in vocab:
                    vocab.append(word)
                    file.write(word + "\n")
        file.write("<unk>\n")
        

def create_word2index_file(saved_folder):
    with open(os.path.join(saved_folder, "vocab.txt"), "r", encoding="utf-8") as file:
        words = [line.strip() for line in file]

    word2index = dict()
    index2word = dict()

    for index, word in enumerate(words):
        word2index[word] = index
        index2word[index] = word

    with open(os.path.join(saved_folder, "word2index.json"), "w", encoding="utf-8") as file:
        json.dump(word2index, file, ensure_ascii=False, indent=4)

    with open(os.path.join(saved_folder, "index2word.json"), "w", encoding="utf-8") as file:
        json.dump(index2word, file, ensure_ascii=False, indent=4)

@dataclass
class LSRow:
    duration: float
    file_path: str
    words: str


def process_line(wav_file, transcript) -> LSRow:
    transcript = transcript.replace("<unk>", "")

    info = read_audio_info(wav_file)
    duration = info.num_frames / info.sample_rate

    return LSRow(
        duration=duration,
        file_path=wav_file,
        words=transcript,
    )
    

if __name__ == "__main__":
    args = parser.parse_args()

    prepare_vinbigdata_vlsp(
        data_folder=args.data_folder,
        saved_folder=args.saved_folder,
        create_vocab=args.create_vocab,
        skip_prep=args.skip_prep
    )



# python C:\paper\Transformer-Transducer-main\linhtinh\vinbigdata_vlsp_prepare.py --data_folder C:\paper\raw_data\vlsp2020_train_set_02 --saved_folder C:\paper\raw_data\VinBigData-VLSP-100h --create_vocab False --skip_prep False