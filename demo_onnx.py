import onnx
import onnxruntime
import numpy as np
from os.path import join as pj
from tqdm import tqdm
import argparse
import re
import json

import transformers
from transformers import AutoTokenizer
    
def tokenize(text: str, tokenizer):
    t = tokenizer(text, padding='max_length', max_length=2048, truncation=True, return_tensors='np')
    input_ids = t['input_ids']
    token_type_ids = t['token_type_ids']
    attention_mask = t['attention_mask']
    return input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()

def remove_html(text: str):
    html_code_pattern = "<\S{1,}>"
    substrings = re.findall(html_code_pattern, text)
    for substring in substrings:
        text = text.replace(substring, '')
    return text
    
def filter_symbols(text: str):
    new_text = ""
    for char in text:
        if char in 'абвгдежзийклмнопрстуфхцчшщъыьэюя ':
            new_text += char
    return new_text
    
def lowercase(text: str):
    return text.lower()   

def predict(input_ids, attention_mask, ort_session):
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids, \
                  ort_session.get_inputs()[1].name: attention_mask}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = np.array(ort_outs)
    ort_outs = np.squeeze(ort_outs, axis=(0,1)).argmax()
    return ort_outs

def preprocess(text: str, tokenizer):
    text = lowercase(text)
    text = remove_html(text)
    text = text.replace('ё', 'е')
    text = filter_symbols(text) 
    input_ids, token_type_ids, attention_mask = tokenize(text, tokenizer)
    input_ids = np.expand_dims(input_ids, 0)
    attention_mask = np.expand_dims(attention_mask, 0)
    return input_ids, attention_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="bert-tiny.onnx")
    parser.add_argument('--tokenizer', type=str, default="bert_tokenizer")
    parser.add_argument('--labels', type=str, default="labels.json")
    args = parser.parse_args()

    with open(args.labels, 'r', encoding='utf-8') as file:
        class_id2label = json.load(file)

    model_path = args.weights
    tokenizer_path = args.tokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(model_path)

    while True:
        text = input('Введите запрос: ')
        input_ids, attention_mask = preprocess(text, tokenizer)
     
        pred = predict(input_ids, attention_mask, ort_session)
        print("Тема запроса:", class_id2label[str(pred)])


