### Training PPO Agent
- First training
    ```
    python main.py
    ```
- If performance is not good enough, try continual training with **smaller learning rate**.
    ```shell
    python main2.py
    ```

### Evaluate and Store the video
- Change your weight path at line 33 in `record.py`
- Run the following instruction
    ```shell
    python record.py
    ```