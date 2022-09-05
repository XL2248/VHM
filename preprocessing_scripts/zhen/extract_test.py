#coding=utf-8
import os, code
from lxml import etree

fpath = "ZH2ENSUM_test.txt"
ori_path = "PART_I.txt"

with open(fpath, 'r') as f, open(ori_path, 'r') as f2, open("test_zh.text", 'w') as f_text, open("test_zh.sum", 'w') as f_en, open("test_en.sum", 'w') as f_zh:
    content = f2.readlines()
    all_sum = f.readlines()
    #selector = etree.HTML(ori_path)
    #code.interact(local=locals())
    start, end = 0, 0
    id_text = {}
    id_sum = {}
    for idx, line in enumerate(content):
        if idx % 8 == 0:
            index = line.strip().split('=')[1].split('>')[0]
            text = content[idx + 5].strip()
            id_text[index] = text

    for idx, line in enumerate(all_sum):
        if idx % 11 == 0:
            tmp_list = []
            index = line.strip().split('=')[1].split('>')[0]
            zh_text = all_sum[idx + 5].strip()
            en_text = all_sum[idx + 8].strip()
            tmp_list.append(en_text)
            tmp_list.append(zh_text)
            id_sum[index] = tmp_list
    #code.interact(local=locals())
    for k, v in id_sum.items():
        if k in id_text:
            text = id_text[k]
            en_sum = v[0]
            ch_sum = v[1]
            f_text.write(text + '\n')
            f_en.write(en_sum + '\n')
            f_zh.write(ch_sum + '\n')
        else:
            print("not exist", k)
#            code.interact(local=locals())
