import os
import matplotlib.pyplot as plt


# settings
plot_title = 'VGG19CSR2-GeneralisedLoss-NWPU Dataset' #'VGG19 + CSRnet-Styled Dilated Regression Layer'

# save dir for plots
save_path = '/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/plots/vgg19csr2'

# data directory
fp = '/home/students/s121md106_04/GeneralizedLoss-Counting-Pytorch-main/checkpoints/vgg19csr2_GLoss_nwpu/0916-195251/train.log'




f = open(fp, 'r')
lines = f.readlines()
# ep = 0
train_losses, train_mse_, train_mae_, val_mse_, val_mae_ = [], [], [], [], []
for line in lines:
    if 'Train, Loss:' in line:
        epoch = int(line.split()[3])
        train_loss = float(line.split()[6].strip(','))
        train_mse = float(line.split()[8])
        train_mae = float(line.split()[10].strip(','))
        train_losses.append(train_loss)
        train_mse_.append(train_mse)
        train_mae_.append(train_mae)
    if 'val Epoch' in line:
        epoch = int(line.split()[4].strip(','))
        val_mse = float(line.split()[6])
        val_mae = float(line.split()[8].strip(','))
        val_mse_.append(val_mse)
        val_mae_.append(val_mae)

# normalise for fit curves for checking for fitness
lst = [train_losses, train_mse_, train_mae_, val_mse_, val_mae_]
labels = ['train_losses', 'train_mse', 'train_mae', 'val_mse', 'val_mae']
lst_norm = [[],[],[],[],[]]

# get data lenght
length = len(train_losses)

i = 0
for sv_lst, loss_lst in zip(lst_norm, lst):
    # print(loss_lst)
    sv_lst = [val/max(loss_lst) for val in loss_lst]
    lst_norm[i]=sv_lst 
    # print(len(sv_lst))
    i+=1
# print(len(lst_norm[1]))

X = list(range(0,length))
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i, label in enumerate(labels):
    if i > 2:
        plt.plot(X, lst[i], label=label, color='C1')
    else:
        plt.plot(X, lst[i], label=label, color='C0')
    print(label)
    plt.title(plot_title)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.savefig(f'{save_path}/{label}.png')
    plt.clf()

# plot over lay of train and val results

plt.plot(X, lst_norm[2], label='train_mae')
plt.plot(X, lst_norm[4], label='val_mae')
plt.title(plot_title + ', Normalised MAE')
plt.xlabel('epoch')
plt.ylabel('mae (norm)')
plt.legend()
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig(os.path.join(save_path, 'train_vs_val_mae_norm.png'))
plt.clf()

plt.plot(X, lst_norm[1], label='train_mse')
plt.plot(X, lst_norm[3], label='val_mse')
plt.title(plot_title + ', Normalised MSE')
plt.xlabel('epoch')
plt.ylabel('mse (norm)')
plt.legend()
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig(os.path.join(save_path, 'train_vs_val_mse_norm.png'))
plt.clf()

plt.plot(X, lst[2], label='train_mae')
plt.plot(X, lst[4], label='val_mae')
plt.title(plot_title)
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend()
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig(os.path.join(save_path, 'train_vs_val_mae.png'))
plt.clf()

plt.plot(X, lst[1], label='train_mse')
plt.plot(X, lst[3], label='val_mse')
plt.title(plot_title)
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig(os.path.join(save_path, 'train_vs_val_mse.png'))
plt.clf()