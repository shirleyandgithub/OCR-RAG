# encoding=utf-8

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "/root/nllb-200-distilled-600M"

# 语言代码
SOURCE_LANG = "zho_Hans"  # 中文
TARGET_LANG = "eng_Latn"  # 乌兹别克语
EXIT_CMD = "exit"
"""
zho_Hans  # 中文
eng_Latn  # 英语
deu_Latn  # 德语
fra_Latn  # 法语
ita_Latn  # 意大利语
spa_Latn  # 西班牙语
por_Latn  # 葡萄牙语
rus_Cyrl  # 俄语
arb_Arab  # 阿拉伯语
jpn_Jpan  # 日语
kor_Hang  # 韩语
uzn_Latn  # 乌兹别克斯坦语
"""

def main():
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    # 手动设置源语言
    tokenizer.src_lang = SOURCE_LANG

    while True:
        # 优化：显示实际的源语言代码（如 [zho_Hans]）
        text = input(f"[{SOURCE_LANG}] : ")
        if text.lower() == EXIT_CMD:
            break
        if not text.strip():
            continue

        # 编码输入
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        # 生成翻译
        target_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_length=128,
            num_beams=5,  # 束搜索, 让模型在每一步生成时考虑多个候选项
            length_penalty=1.2,  # 鼓励生成稍长的句子
            no_repeat_ngram_size=2  # 防止重复（比如“的，的，的”），适用于长句或低资源语言
        )

        # 解码输出
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[{TARGET_LANG}] : {result}\n")

if __name__ == "__main__":
    main()



