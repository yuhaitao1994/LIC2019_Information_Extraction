# -*- coding: utf-8 -*-
from tqdm import tqdm

with open('test.txt', 'r') as f:
    index = 0
    text = f.readline().strip().split()[0]
    while text:
        index += 1
        try:
            text_next = f.readline().strip().split()[0]
            while text == text_next:
                text_next = f.readline().strip().split()[0]
            text = text_next
        except:
            break
    print(index)
