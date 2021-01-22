import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

#                   0     1     2    3     4     5     6     7     8     9    10
sens_n = np.array([1.0, 1.0, 1.0, 1.0, 0.98, 0.92, 0.78, 0.0, 0.0, 0.0, 0.0]) #recall
spec_n = np.array([0.0, 0.0, 0.0, 0.0, 0.67, 0.94, 0.96, 0.98, 1.0, 1.0, 1.0])
auc_n = metrics.auc(spec_n,sens_n)

sens_a = np.array([1.0, 1.0, 1.0, 1.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
spec_a = np.array([0.0, 0.0, 0.0, 0.0, 0.27, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
auc_a = metrics.auc(spec_a,sens_a)

sens_w = np.array([1.0, 1.0, 1.0, 1.0, 0.96, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
spec_w = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.98, 0.98, 1.0, 1.0, 1.0, 1.0])
auc_w = metrics.auc(spec_w,sens_w)

sens_m = np.array([1.0, 1.0, 1.0, 1.0, 0.93, 0.84, 0.75, 0.0, 0.0, 0.0, 0.0])
spec_m = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.85, 0.9, 0.94, 1.0, 1.0, 1.0])
auc_m = metrics.auc(spec_m,sens_m)

sens_p = np.array([1.0, 1.0, 1.0, 1.0, 0.88, 0.78, 0.67, 0.0, 0.0, 0.0, 0.0])
spec_p = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.92, 0.9, 0.94, 1.0, 1.0, 1.0])
auc_p = metrics.auc(spec_p,sens_p)

sens_S = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 1.0, 0.0, 0.0, 0.0, 0.0])
spec_S = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
auc_S = metrics.auc(spec_S,sens_S)

sens_O = np.array([1.0, 1.0, 1.0, 1.0, 0.97, 0.84, 0.5, 0.0, 0.0, 0.0, 0.0])
spec_O = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.91, 0.98, 0.98, 0.99, 0.99, 1.0])
auc_O = metrics.auc(spec_O,sens_O)

plt.title('AUROC curve',fontsize = 20)
plt.xlabel('False Positive Rate(1 - Specificity)', fontsize = 15)
plt.ylabel('True Positive Rate(Sensitivity)', fontsize = 15)

plt.plot(1-spec_n, sens_n, 'b', label = '1Normal (AUC = %0.2f)' % auc_n)
plt.plot(1-spec_a, sens_a, 'g', label = '2Adenoma (AUC = %0.2f)' % auc_a)
plt.plot(1-spec_w, sens_w, 'r', label = '3Adenocarcinoma,WD (AUC = %0.2f)' % auc_w)
plt.plot(1-spec_m, sens_m, 'c', label = '4Adenocarcinoma,MD (AUC = %0.2f)' % auc_m)
plt.plot(1-spec_p, sens_p, 'k', label = '5Adenocarcinoma,PD (AUC = %0.2f)' % auc_p)
plt.plot(1-spec_m, sens_m, 'c', label = '6Adenocarcinoma,MD (AUC = %0.2f)' % auc_S)
plt.plot(1-spec_p, sens_p, 'k', label = '7Adenocarcinoma,PD (AUC = %0.2f)' % auc_O)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(fontsize = 12, loc='lower right')
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.show()

# densenet
# sens_n = np.array([1.0, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.94, 0.96, 0.0]) #recall
# spec_n = np.array([0.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 1.0])
# auc_n = metrics.auc(spec_n,sens_n)
#
# sens_a = np.array([1.0, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.0])
# spec_a = np.array([0.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 1.0])
# auc_a = metrics.auc(spec_a,sens_a)
#
# sens_w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# spec_w = np.array([0.0, 0.96, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97, 0.98, 0.98, 1.0])
# auc_w = metrics.auc(spec_w,sens_w)
#
# sens_m = np.array([1.0, 0.83, 0.83, 0.83, 0.83, 0.83, 0.88, 0.89, 0.90, 0.92, 0.0])
# spec_m = np.array([0.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0])
# auc_m = metrics.auc(spec_m,sens_m)
#
# sens_p = np.array([1.0, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.98, 0.99, 1.00, 0.0])
# spec_p = np.array([0.0, 0.99, 0.99, 0.99, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00, 1.0])
# auc_p = metrics.auc(spec_p,sens_p)

# inception v3
# sens_n = np.array([1.0, 0.85, 0.85, 0.85, 0.85, 0.85, 0.88, 0.88, 0.91, 0.91, 0.0]) #recall
# spec_n = np.array([0.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 1.0])
# auc_n = metrics.auc(spec_n,sens_n)
#
# sens_a = np.array([1.0, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.83, 0.85, 0.85, 0.0])
# spec_a = np.array([0.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 1.0])
# auc_a = metrics.auc(spec_a,sens_a)
#
# sens_w = np.array([1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.0])
# spec_w = np.array([0.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0])
# auc_w = metrics.auc(spec_w,sens_w)
#
# sens_m = np.array([1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.0])
# spec_m = np.array([0.0, 0.95, 0.95, 0.95, 0.95, 0.95, 0.96, 0.96, 0.97, 0.98, 1.0])
# auc_m = metrics.auc(spec_m,sens_m)
#
# sens_p = np.array([1.0, 0.92, 0.92, 0.92, 0.92, 0.92, 0.93, 0.94, 0.97, 0.98, 0.0])
# spec_p = np.array([0.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0])
# auc_p = metrics.auc(spec_p,sens_p)

# resnet
# sens_n = np.array([1.0, 0.92, 0.92, 0.92, 0.92, 0.93, 0.93, 0.94, 0.95, 0.96, 0.0]) #recall
# spec_n = np.array([0.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 1.0])
# auc_n = metrics.auc(spec_n,sens_n)
#
# sens_a = np.array([1.0, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.89, 0.89, 0.0])
# spec_a = np.array([0.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 1.0])
# auc_a = metrics.auc(spec_a,sens_a)
#
# sens_w = np.array([1.0, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0])
# spec_w = np.array([0.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0])
# auc_w = metrics.auc(spec_w,sens_w)
#
# sens_m = np.array([1.0, 0.94, 0.94, 0.94, 0.94, 0.95, 0.95, 0.96, 0.97, 0.98, 0.0])
# spec_m = np.array([0.0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 1.0])
# auc_m = metrics.auc(spec_m,sens_m)
#
# sens_p = np.array([1.0, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.0])
# spec_p = np.array([0.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0])
# auc_p = metrics.auc(spec_p,sens_p)