train_path = "/home/quhongling/dataset/mix_2/train/"
f = open(r'/home/quhongling/experiments/Network-test/train_dataset_path.txt', 'w')
for i in range(3780):
    j = i + 1
    f.write(train_path + "mix/mix_" + str(j) + ".wav " +
            train_path + "target/target_" + str(j) + ".wav " +
            train_path + "refer/refer_" + str(j) + ".wav\n")
f.close()
print("mix2 训练集文件写入完毕")

dev_path = "/home/quhongling/dataset/mix_2/dev/"
f = open(r'/home/quhongling/experiments/Network-test/dev_dataset_path.txt', 'w')
for i in range(1260):
    j = i + 1
    f.write(dev_path + "mix/mix_" + str(j) + ".wav " +
            dev_path + "target/target_" + str(j) + ".wav " +
            dev_path + "refer/refer_" + str(j) + ".wav\n")
f.close()
print("mix2 验证集文件写入完毕")
