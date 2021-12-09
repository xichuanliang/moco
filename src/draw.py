import matplotlib.pyplot as plt

input = open('../experiments/2021-09-19_10-39-12/trainlogs.txt')
input_change = open('../experiments/2021-11-11_09-22-22/trainlogs.txt')

loss = []
loss_change = []

x = range(1,101)

for line in input:
    line = line.split()
    if '[valid]' in line:
        loss.append(float(line[4].replace(',','')))

for line in input_change:
    line = line.split()
    if '[valid]' in line:
        loss_change.append(float(line[4].replace(',','')))


plt.plot(x, loss, label='movo' )
plt.plot(x,loss_change,'r', label = 'movo_change')
plt.legend()
plt.savefig('../experiments/2021-11-11_09-22-22/change.jpg')
plt.show()
