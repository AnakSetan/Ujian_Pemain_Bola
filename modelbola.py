import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
dfFifa = df[['Name','Age','Overall','Potential']]
# print(dfFifa)
dftarget = dfFifa[dfFifa['Age'] <= 25][dfFifa['Overall'] >= 80][dfFifa['Potential'] >= 80]
# print(dftarget)
dfnontarget = dfFifa.drop(dftarget.index)
# print(dfnontarget) 

fig = plt.figure('Fifa', figsize = (13,6))

ax = plt.subplot(121)

xt1 = (dftarget['Age'].values).reshape(-1, 1)
# print(xt1)
yt1 = dftarget['Overall'].values
# print(yt1)
plt.scatter(
    xt1,
    yt1,
    color = 'g',
    s = 5,
    label = 'Target'
)


xt1 = (dfnontarget['Age'].values).reshape(-1, 1)
# print(xt1)
yt1 = dfnontarget['Overall'].values
# print(yt1)
plt.scatter(
    xt1,
    yt1,
    color = 'r',
    s = 5,
    label = 'Non-Target'
)

ax.legend()
ax.grid(True)
ax.set_xlabel('Age')
ax.set_ylabel('Overall')
ax.set_title('Age vs Overall')
ax.legend()
ax.grid(True)


bx = plt.subplot(122)

xt2 = (dftarget['Age'].values).reshape(-1, 1)
yt2 = dftarget['Potential'].values
plt.scatter(
    xt2,
    yt2,
    color = 'g',
    s = 5,
    label = 'Target'
)

xt2 = (dfnontarget['Age'].values).reshape(-1, 1)
yt2 = dfnontarget['Potential'].values
plt.scatter(
    xt2,
    yt2,
    color = 'r',
    s = 5,
    label = 'Non-Target'
)

bx.set_xlabel('Age')
bx.set_ylabel('Potential')
bx.set_title('Age vs Potential')
bx.legend()
bx.grid(True)

plt.show()
