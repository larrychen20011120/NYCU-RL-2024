#!/bin/bash
# 使用 for 迴圈執行 20 次
for i in {201..300}
# 取得當前時間的時間戳
do
    # 執行 ./2048_evaluate，並將結果輸出到 result_$i.txt 檔案中
    ./evaluate > "find_seed_result/result_$i.txt" &
    echo "Execution $i completed and saved to result_$i.txt"
done

wait
echo "All executions completed."
