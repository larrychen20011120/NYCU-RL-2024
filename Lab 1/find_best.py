import os
import re


def find_highest_2048(folder_path):
    highest_2048_file = None
    highest_2048_value = 0

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # 讀取檔案內容
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                # 使用正則表達式找到"2048%"的行並提取數值
                match = re.search(r"2048\s+([0-9.]+)%", content)
                if match:
                    value = float(match.group(1))
                    if value > highest_2048_value:
                        highest_2048_value = value
                        highest_2048_file = filename

    return highest_2048_file, highest_2048_value


# 使用範例
folder_path = "find_seed_result"  # 替換成你的資料夾路徑
file, value = find_highest_2048(folder_path)

if file:
    print(f"2048%數值最高的檔案是: {file}，數值為: {value}%")
else:
    print("沒有找到2048%的數值")
