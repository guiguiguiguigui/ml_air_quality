from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc(y_test, y_predicted):
    fpr, tpr, _ = roc_curve(y_test, y_predicted)
    roc_auc = auc(fpr, tpr)
    print "AUROC score: %.2f" % roc_auc

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve of Prediction')
    plt.legend(loc="lower right")
    plt.show()
    