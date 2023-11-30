from matplotlib import pyplot as plt

filename = "bce 1000"

filename_train = 'output/' + filename + '/train/train_error.txt'
filename_valid = 'output/' + filename + '/valid/valid_error.txt'

losses_train = []
train_iou_list = []
train_dice_list = []
train_sens_list = []
train_spec_list = []
train_aoc_list = []
train_haus_list = []

losses_valid = []
valid_iou_list = []
valid_dice_list = []
valid_sens_list = []
valid_spec_list = []
valid_aoc_list = []
valid_haus_list = []

i = 0

f = open(filename_train, "r")
for line in f:
  i += 1
  info = line.split()
  if len(info) > 3:
    if info[0] != 'epoch':
        continue
    loss = float(info[4])
    iou = float(info[7][:6])
    dice = float(info[10][:6])
    sens = float(info[13][:6])
    spec = float(info[16][:6])
    aoc = float(info[19][:6])
    haus = float(info[22][:6])

    train_iou_list.append(iou)
    train_dice_list.append(dice)
    train_sens_list.append(sens)
    train_spec_list.append(spec)
    train_aoc_list.append(aoc)
    train_haus_list.append(haus)

    losses_train.append(loss)
  
i = 0
f = open(filename_valid, "r")
for line in f:
  i += 1
  info = line.split()
  if len(info) > 3:
      if info[0] != 'epoch':
        continue
      loss = float(info[4])

      iou = float(info[7][:6])
      dice = float(info[10][:6])
      sens = float(info[13][:6])
      spec = float(info[16][:6])
      aoc = float(info[19][:6])
      haus = float(info[22][:6])

      #if i % 10:
      #  continue
      valid_iou_list.append(iou)
      valid_dice_list.append(dice)
      valid_sens_list.append(sens)
      valid_spec_list.append(spec)
      valid_aoc_list.append(aoc)
      valid_haus_list.append(haus)
      losses_valid.append(loss)                                                                                                                                                                                                                         

# plt.ylim(0, 0.5)
plt.plot(range(len(losses_train)), losses_train, label='train')
plt.plot(range(len(losses_valid)), losses_valid, label='valid')
#plt.plot([i*10 for i in range(len(losses_valid))], losses_valid, label='valid')
plt.xlabel("Epoha")
plt.ylabel("loss")
plt.title("Dice Loss")
plt.legend(loc="upper left")
plt.show()

plt.plot(range(len(train_iou_list)), train_iou_list, label='train')
plt.plot(range(len(valid_iou_list)), valid_iou_list, label='valid')
# plt.plot([i*10 for i in range(len(valid_iou_list))], valid_iou_list, label='valid')
plt.xlabel("Epoha")
plt.ylabel("loss")
plt.title("IoU")
plt.legend(loc="upper left")
plt.show()

plt.plot(range(len(train_dice_list)), train_dice_list, label='train')
plt.plot(range(len(valid_dice_list)), valid_dice_list, label='valid')
# plt.plot([i*10 for i in range(len(valid_dice_list))], valid_dice_list, label='valid')
plt.xlabel("Epoha")
plt.ylabel("loss")
plt.title("Dice score")
plt.legend(loc="upper left")
plt.show()

plt.plot(range(len(train_sens_list)), train_sens_list, label='train')
plt.plot(range(len(valid_sens_list)), valid_sens_list, label='valid')
# plt.plot([i*10 for i in range(len(valid_sens_list))], valid_sens_list, label='valid')
plt.xlabel("Epoha")
plt.ylabel("loss")
plt.title("Sensitivity")
plt.legend(loc="upper left")
plt.show()

plt.plot(range(len(train_spec_list)), train_spec_list, label='train')
plt.plot(range(len(valid_spec_list)), valid_spec_list, label='valid')
#plt.plot([i*10 for i in range(len(valid_spec_list))], valid_spec_list, label='valid')
plt.xlabel("Epoha")
plt.ylabel("loss")
plt.title("Specificity")
plt.legend(loc="upper left")
plt.show()

plt.plot(range(len(train_aoc_list)), train_aoc_list, label='train')
plt.plot(range(len(valid_aoc_list)), valid_aoc_list, label='valid')
#plt.plot([i*10 for i in range(len(valid_aoc_list))], valid_aoc_list, label='valid')
plt.xlabel("Epoha")
plt.ylabel("loss")
plt.title("AOC")
plt.legend(loc="upper left")
plt.show()

plt.plot(range(len(train_haus_list)), train_haus_list, label='train')
plt.plot(range(len(valid_haus_list)), valid_haus_list, label='valid')
# plt.plot([i*10 for i in range(len(valid_haus_list))], valid_haus_list, label='valid')
plt.xlabel("Epoha")
plt.ylabel("loss")
plt.title("Hausdorff")
plt.legend(loc="upper left")
plt.show()