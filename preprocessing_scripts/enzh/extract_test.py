#coding=utf-8
import os, code
fpath = "EN2ZHSUM_test.txt"

with open(fpath, 'r') as f, open("test_en.text", 'w') as f_text, open("test_en.sum", 'w') as f_en, open("test_zh.sum", 'w') as f_zh:
    content = f.readlines()
    start, end = 0, 0
    for idx, line in enumerate(content):
        if line.strip() == '<Article>':
            text = content[idx + 1].strip().lower()
            start = idx + 4
        if line.strip() == '</doc>':
            end = idx
            en_sum = ''
            ch_sum = ''
            for index in range(end - start):
                if start + index * 12 < end:
                    if index == 0:
                        en_sum = content[start + index * 12].strip().lower()
                        ch_sum = content[start + index * 12 + 6].strip()
                    else:
                        en_sum += ' ' + content[start + index * 12].strip().lower()
                        ch_sum += ' ' + content[start + index * 12 + 6].strip()
            f_text.write(text + '\n')
            f_en.write(en_sum + '\n')
            f_zh.write(ch_sum.decode('utf8').encode('utf8') + '\n')

