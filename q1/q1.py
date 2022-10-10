# Sahil Singh Rathore
# B20227
# Mob No: 9559176048

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import statistics


def prob(x, u, v):
    d = x-u
    num = np.exp(-(d*d)/(2*v))
    den = np.sqrt(2*np.pi*v)

    p = num/den
    return p


mel_train = pd.read_csv(
    "q1/data/Segment2_MelEnergy.csv", header=None)[0].to_numpy()
mel_test = pd.read_csv("q1/data/Segment3_MelEnergy.csv",
                       header=None)[0].to_numpy()

ste_train = pd.read_csv("q1/data/Segment2_STEnergy.csv",
                        header=None)[0].to_numpy()
ste_test = pd.read_csv("q1/data/Segment3_STEnergy.csv",
                       header=None)[0].to_numpy()

class_train = pd.read_csv(
    "q1/data/Segment2_VAD_GT.csv", header=None)[0].to_numpy()
class_test = pd.read_csv("q1/data/Segment3_VAD_GT.csv",
                         header=None)[0].to_numpy()


ste_sound = []
mel_sound = []

for i in range(len(class_train)):
    if class_train[i] == 1:
        ste_sound.append(ste_train[i])
        mel_sound.append(mel_train[i])


mel_sound_mean = sum(mel_sound)/len(mel_sound)
ste_sound_mean = sum(ste_sound)/len(ste_sound)

mel_sound_var = statistics.variance(mel_sound)
ste_sound_var = statistics.variance(ste_sound)


tp_mel = []
fp_mel = []
tp_ste = []
fp_ste = []
for p in np.arange(0, 3, 0.01):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(class_test)):
        x = prob(mel_test[i], mel_sound_mean, mel_sound_var)
        if x > p:
            if class_test[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if class_test[i] == 1:
                fn += 1
            else:
                tn += 1

    tp_mel.append(tp/(tp+fn))
    fp_mel.append(fp/(tn+fp))

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(class_test)):
        x = prob(ste_test[i], ste_sound_mean, ste_sound_var)
        if x > p:
            if class_test[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if class_test[i] == 1:
                fn += 1
            else:
                tn += 1
    tp_ste.append(tp/(tp+fn))
    fp_ste.append(fp/(tn+fp))


plt.plot(fp_mel, tp_mel, color='green')
plt.plot(fp_ste, tp_ste, color='red')
plt.title("ROC Curve")
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.legend(["MEL Energy", "STE Energy"])
plt.show()
