import json
with open('datasets/marc_ja-v1.1/train-v1.0.json', 'r') as f:
    # 各行を個別に読み取る
    train_data = []
    for line in f:
        data = json.loads(line)  # 各行を辞書として読み込む
        train_data.append(data)
with open('datasets/marc_ja-v1.1/valid-v1.0.json', 'r') as f:
    # 各行を個別に読み取る
    valid_data = []
    for line in f:
        data = json.loads(line)  # 各行を辞書として読み込む
        valid_data.append(data)

import openai
limit = 3
result = []
for data in valid_data[:limit]:
    # https://arxiv.org/abs/2302.10198
    prompt = f'Q: For the sentence: "{data["sentence"]}", is the sentiment in this sentence positive or negative? A: The answer (positive or negative) is:'
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                    ],
                temperature=0,
            )['choices'][0]['message']['content']
    
    data['ChatGPT-3.5-Turbo_Zhong'] = response
    

    # https://fintan.jp/page/9126/
    prompt = f'製品レビューをnegativeかpositiveのいずれかのセンチメントに分類してください。出力は小文字化してください。製品レビュー: {data["sentence"]} センチメント:  '
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                    ],
                temperature=0,
            )['choices'][0]['message']['content']
    data['ChatGPT-3.5-Turbo_fintan'] = response

    # https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/
    prompt = f"""Your task is classifying the sentiment of the sentence. The sentiment is either positive or negative. 
Use the following format.
Sentence: 
```
sentence here
```
Sentiment:
```
positive or negative
```

Sentence: 
```
{data["sentence"]}
```
Sentiment:
    """
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                    ],
                temperature=0,
            )['choices'][0]['message']['content']
    data['ChatGPT-3.5-Turbo_andrew'] = response
    result.append(data)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")

for data in result:
    prompt = f'Q: For the sentence: "{data["sentence"]}", is the sentiment in this sentence positive or negative? A: The answer (positive or negative) is:'
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    data['open-calm-7b_Zhong'] = output[len(prompt):]

    prompt = f'製品レビューをnegativeかpositiveのいずれかのセンチメントに分類してください。出力は小文字化してください。製品レビュー: {data["sentence"]} センチメント:  '
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    data['open-calm-7b_fintan'] = output[len(prompt):]

    prompt = f"""Your task is classifying the sentiment of the sentence. The sentiment is either positive or negative. 
Use the following format.
Sentence: 
```
sentence here
```
Sentiment:
```
positive or negative
```

Sentence: 
```
{data["sentence"]}
```
Sentiment:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    data['open-calm-7b_andrew'] = output[len(prompt):]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b")

if torch.cuda.is_available():
    model = model.to("cuda")

for data in result:
    prompt = f'Q: For the sentence: "{data["sentence"]}", is the sentiment in this sentence positive or negative? A: The answer (positive or negative) is:'
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    data['rinna-japanese-gpt-neox-3.6b_Zhong'] = output[len(prompt):]

    prompt = f'製品レビューをnegativeかpositiveのいずれかのセンチメントに分類してください。出力は小文字化してください。製品レビュー: {data["sentence"]} センチメント:  '
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    data['rinna-japanese-gpt-neox-3.6b_fintan'] = output[len(prompt):]

    prompt = f"""Your task is classifying the sentiment of the sentence. The sentiment is either positive or negative. 
Use the following format.
Sentence: 
```
sentence here
```
Sentiment:
```
positive or negative
```

Sentence: 
```
{data["sentence"]}
```
Sentiment:
    """
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    data['rinna-japanese-gpt-neox-3.6b_andrew'] = output[len(prompt):]


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")

if torch.cuda.is_available():
    model = model.to("cuda")

for data in result:
    prompt = f'Q: For the sentence: "{data["sentence"]}", is the sentiment in this sentence positive or negative? A: The answer (positive or negative) is:'
    rinna_prompt = [
        {
            "speaker": "ユーザー",
            "text": prompt
        },
    ]
    prompt = [
        f"{uttr['speaker']}: {uttr['text']}"
        for uttr in rinna_prompt
    ]
    prompt = "<NL>".join(prompt)
    prompt = (
        prompt
        + "<NL>"
        + "システム: "
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=512,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    data['rinna-japanese-gpt-neox-3.6b-instruction-sft_Zhong'] = output

    prompt = f'製品レビューをnegativeかpositiveのいずれかのセンチメントに分類してください。出力は小文字化してください。製品レビュー: {data["sentence"]} センチメント:  '
    rinna_prompt = [
        {
            "speaker": "ユーザー",
            "text": prompt
        },
    ]
    prompt = [
        f"{uttr['speaker']}: {uttr['text']}"
        for uttr in rinna_prompt
    ]
    prompt = "<NL>".join(prompt)
    prompt = (
        prompt
        + "<NL>"
        + "システム: "
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=512,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    data['rinna-japanese-gpt-neox-3.6b-instruction-sft_fintan'] = output

    prompt = f"""Your task is classifying the sentiment of the sentence. The sentiment is either positive or negative. 
Use the following format.
Sentence: 
```
sentence here
```
Sentiment:
```
positive or negative
```

Sentence: 
```
{data["sentence"]}
```
Sentiment:
    """
    rinna_prompt = [
        {
            "speaker": "ユーザー",
            "text": prompt
        },
    ]
    prompt = [
        f"{uttr['speaker']}: {uttr['text']}"
        for uttr in rinna_prompt
    ]
    prompt = "<NL>".join(prompt)
    prompt = (
        prompt
        + "<NL>"
        + "システム: "
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=512,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    data['rinna-japanese-gpt-neox-3.6b-instruction-sft_andrew'] = output

import pickle
with open('result.pkl', 'wb') as f:
    pickle.dump(result, f)

with open('result.pkl', 'rb') as f:
    loaded_result = pickle.load(f)

print(loaded_result)