import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_json('newdata.json')

stars = [0] * 5

stars[0] = data[data['overall'] == 1].shape[0]
stars[1] = data[data['overall'] == 2].shape[0]
stars[2] = data[data['overall'] == 3].shape[0]
stars[3] = data[data['overall'] == 4].shape[0]
stars[4] = data[data['overall'] == 5].shape[0]

print(f"Breakdown:\n1: {stars[0]}\n2: {stars[1]}\n3: {stars[2]}\n4: {stars[3]}\n5: {stars[4]}")

# Plotting
'''
plt.bar(range(1, len(stars) + 1), stars, color='skyblue', edgecolor='black')
plt.title('Star Rating Breakdown')
plt.xlabel('Star Rating')
plt.ylabel('Number of Reviews')
plt.xticks(range(1, len(stars) + 1))
plt.show()
'''

print(f"Percent of 5 stars to rest of model is {stars[4]/data.shape[0]}")


