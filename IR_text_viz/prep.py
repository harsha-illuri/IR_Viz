import os
import pandas as pd
import numpy as np
from IR_text_viz.class_prim import *
pd.set_option('display.expand_frame_repr', False)
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from IR_text_viz.lstm_model import *



class preprocess:
    def __init__(self, path, process_path = 'proc'):
        self.path = path
        self.process_path = process_path

    def removeShortSentences(self, row):
        if len(row) > 20:
            return row
        else:
            return

    def getClass(self,row):
        for key, vals in error_map.items():
            for v in vals:
                if v in row:
                    return key
        return None

    def process_raw(self):
        # os.remove('raw_data.h5')
        main_df = pd.DataFrame()
        if not os.path.isdir(self.process_path):
            os.mkdir(self.process_path)
        for file_path in os.listdir(self.path):
            if '.txt' in file_path:
                file = file_path
                file_path = self.path+'/'+file_path
                print(file_path)
                f = open(file_path, encoding="windows-1252")
                df = pd.DataFrame()
                sent = []
                lines = []
                pos = []
                end_pos = []
                for line_no, line in enumerate(f.readlines()):
                    p = 0
                    sent_array = sent_tokenize(line)
                    for s in sent_array:
                        lines.append(line_no)
                        sent.append(s)
                        pos.append(p)
                        p += len(s)
                        end_pos.append(p)
                df['text'] = sent
                df['line'] = lines
                df['start_pos'] = pos
                df['end_pos'] = end_pos
                df['file'] = file
                main_df = main_df.append(df)
                f.close()
        main_df.to_hdf('raw_data.h5',key='raw',min_itemsize={'text': 4096})
        # print(main_df)

    def getTrainingData(self):
        df = pd.read_hdf('raw_data_bkp.h5', key='raw')
        df['text'] = df['text'].apply(self.removeShortSentences)
        df.dropna(subset=['text'],inplace=True)
        df['class'] = df['text'].apply(self.getClass)
        df.dropna(subset=['class'], inplace=True)
        df.to_hdf('training_data.h5', key='key', format='t')

    def getTestData(self):
        df = pd.read_hdf('raw_data.h5', key='raw')
        df['text'] = df['text'].apply(self.removeShortSentences)
        df.dropna(subset=['text'], inplace=True)
        df['class'] = df['text'].apply(self.getClass)
        df = df[df['class'].isnull()]
        df.to_hdf('test_data.h5', key='key', format='t')



# p = preprocess('data')
# p.getTrainingData()
# p.getTestData()

