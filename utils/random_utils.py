import random

# 随机生成两个数，每个数可以是 -1 或 1
for i in range(100):
    num1 = random.choice([0, 1])
    num2 = random.choice([0, 1])

    print("随机生成的两个数：", num1, num2)