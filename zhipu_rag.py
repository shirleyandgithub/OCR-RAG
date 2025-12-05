# encoding=utf-8

import os
import requests

def search_local_docs():
    txt_dir = "/root/"
    keywords = ["电力规划", "国际经验"]
    search_results = []

    # 遍历RAG知识库
    for i in range(1, 4):
        filename = f"Text{i}.txt"
        file_path = os.path.join(txt_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relevant_lines = [line.strip() for line in content.split("\n")
                              if any(keyword in line for keyword in keywords)]

            if relevant_lines:
                search_results.append(f"【{filename}】\n" + "\n".join(relevant_lines))
            else:
                search_results.append(f"【{filename}】未找到相关内容")

        except Exception as e:
            search_results.append(f"【{filename}】读取错误：{str(e)}")

    return "\n\n".join(search_results)

def call_zhipu_llm(prompt):
    # 智谱API的key和url
    api_key = "********"
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "glm-4",
        "messages": [
            {"role": "system", "content": "仅使用提供的本地文件内容回答，每个结论标注来自哪个Text文件，不添加外部信息。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API调用失败：{str(e)}，响应内容：{response.text if 'response' in locals() else '无'}"

def main():
    # 检索本地文件
    search_result = search_local_docs()
    # 构建问题
    user_question = f"""
    基于以下本地文件内容，回答问题：
    {search_result}

    翻译内容："为满足乌兹别克斯坦共和国日益增长的需求，并借鉴先进的国际经验以及世界电力工业发展的现代趋势，制定了2020-2030年乌兹别克斯坦电力发展规划"。
    1. "借鉴先进国际经验"具体指参考了哪个国家的经验？
    2. 该规划与这个国家有哪些对应的合作内容？
    """

    # 调用智谱LLM
    answer = call_zhipu_llm(user_question)

    # 输出结果
    print(answer)

if __name__ == "__main__":
    main()



