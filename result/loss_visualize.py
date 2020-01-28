#%%
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
# plt.style.use('default')
losses = np.loadtxt('Seq2Seq_loss_0121_225605.txt')
plt.figure(figsize=(6,4))
# plt.ylim((0.15, 0.28))

plt.xlabel('epoch')
plt.ylabel('loss')
# plt.title('losses of train and valid set')
for i in range (0, len(losses[0])-1):
    losses[0][i] = losses[0][i+1]
print(losses[0])
plt.plot(losses[0])
plt.plot(losses[1])

# find the the lowest valid loss and mark it
min_idx = losses[1].argmin()
min_loss = losses[1][min_idx]
plt.annotate('epoch: {}\nmin loss: {:.4f}'.format(min_idx,min_loss),
             xy=(min_idx, min_loss),
             xycoords='data',
             xytext=(-80, +50),
             textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2"))

plt.xticks(np.arange(0,51,10))
# plt.xticks(list(range(0,101,10)))
# plt.plot(losses[0,40:])
# plt.plot(losses[1,40:])


plt.legend(['Train loss','Valid loss'])
plt.grid(True)
plt.show()

plt.savefig('best_loss.png',bbox_inches = 'tight')


#%%


#%%
