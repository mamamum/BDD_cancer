import numpy as np

data = np.load('might.npz')

print(data.files)

model_name = data['model_name']
y_1 = data['y']
S98 = data['S98']
posterior_arr = data['posterior_arr']
MI = data['MI']
pAUC = data['pAUC']
hd = data['hd']

np.set_printoptions(precision=3, suppress=True)

print("Model Name:", model_name)
# print("y_1:", y_1)
print("S98:", S98)
# print("Posterior Array:", posterior_arr)
print("Mutual Information:", MI)
print("Partial area under ROC:", pAUC)
print("Hellinger Distance:", hd)
