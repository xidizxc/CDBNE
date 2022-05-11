import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics

def cluster_acc(truelabel, predlabel):
    truelabel = truelabel - np.min(truelabel)

    l1 = list(set(truelabel))
    numclass1 = len(l1)

    l2 = list(set(predlabel))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                predlabel[ind] = i
                ind += 1

    l2 = list(set(predlabel))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(truelabel) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if predlabel[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(predlabel))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(predlabel) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(truelabel, new_predict)
    f1_macro = metrics.f1_score(truelabel, new_predict, average="macro")
    return acc, f1_macro


def eva(truelabel, predlabel, epoch=0):
    acc, f1 = cluster_acc(truelabel, predlabel)
    nmi = nmi_score(truelabel, predlabel, average_method="arithmetic")
    ari = ari_score(truelabel, predlabel)
    print(f"epoch {epoch}: nmi {nmi:.4f}, ari {ari:.4f}, acc {acc:.4f}, f1 {f1:.4f}")
    return acc, nmi, ari, f1

