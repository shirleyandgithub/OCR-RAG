# encoding=utf-8

import cv2
import pytesseract
from PIL import Image

def clean_uzbek_text(raw_text):
    # 按行分割 -> 过滤空行 -> 初步去重
    lines = [line.strip().replace("  ", " ") for line in raw_text.split("\n") if line.strip()]
    clean_lines = []

    for line in lines:
        # 过滤重复字符占比超40%的行（收紧阈值）
        max_char_count = max([line.count(c) for c in set(line)]) if line else 0
        repeat_ratio = max_char_count / len(line) if line else 0
        if repeat_ratio > 0.4:
            continue

        # 过滤包含连续3个及以上相同字符的片段
        has_long_repeat = False
        for i in range(len(line) - 2):
            if line[i] == line[i + 1] == line[i + 2]:
                has_long_repeat = True
                break
        if has_long_repeat:
            continue

        # 补充干扰词库(去除非乌兹别克语常用词)
        noise_words = ["alla", "aaa", "jaaa", "iii"]  # 常见干扰词
        has_noise = any(noise in line.lower() for noise in noise_words)
        if has_noise:
            # 若含干扰词，尝试分割保留有效部分（如"xxx maqsadida" -> 保留"maqsadida"）
            parts = line.split()
            valid_parts = [p for p in parts if not any(noise in p.lower() for noise in noise_words)]
            line = " ".join(valid_parts)
            # 分割后若只剩空或短片段，直接过滤
            if len(line) < 5:
                continue

        # 保留乌兹别克语核心字符（排除非目标字符）
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' 0123456789,-.")
        line = "".join([c for c in line if c in valid_chars])

        # 最终过滤短文本
        if len(line) >= 5:
            clean_lines.append(line)

    # 拼接有效文本 -> 最终去重空格
    return " ".join(clean_lines).replace("  ", " ")

def ocr_uzbek_image(image_path):
    # 图片读取与增强（加重对比度，减少噪点干扰）
    img = cv2.imread(image_path)
    if img is None:
        return f"[错误] 图片 {image_path} 不存在"

    # 灰度化 -> 高斯降噪 -> 二值化
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_img = cv2.GaussianBlur(gray_img, (3, 3), 0)  # 新增：高斯降噪
    _, binary_img = cv2.threshold(denoised_img, 140, 255, cv2.THRESH_BINARY_INV)  # 调整阈值

    # Tesseract识别（指定乌兹别克语）
    raw_ocr_result = pytesseract.image_to_string(Image.fromarray(binary_img), lang="uzb")

    # 增强版文本清洗
    clean_ocr_result = clean_uzbek_text(raw_ocr_result)

    # 结果验证
    if not clean_ocr_result:
        return "[错误] 未识别到有效乌兹别克语文本"

    return clean_ocr_result

if __name__ == "__main__":
    TARGET_IMAGE_PATH = "/root/uz.png"
    final_text = ocr_uzbek_image(TARGET_IMAGE_PATH)
    print(f"OCR识别结果（乌兹别克语）\n{final_text}")



