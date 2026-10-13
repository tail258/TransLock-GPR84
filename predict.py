"""推理入口：返回本次输入的全部准入评分图。"""
from src.submission import screen_main

if __name__ == "__main__":
    screen_main(predict_only=True)
