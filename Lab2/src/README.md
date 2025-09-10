### Training Different models
- DQN
    ```shell
    python train_dqn.py
    ```
- DDQN
    ```shell
    python train_ddqn.py
    ```
- DuelingDQN
    ```shell
    python train_dueldqn.py
    ```
- Under Parallel Environment
    ```shell
    python train_parallel.py
    ```

### Evaluation

```shell
python evaluate.py \
--weight "YOUR_MODEL_PATH" \
--model "DQN/DDQN/DuelDQN: your agent's algorithm" \
--game "MsPacman/Enduro: which game to play" \
--mode "video/print: store the video or print out the result"
```