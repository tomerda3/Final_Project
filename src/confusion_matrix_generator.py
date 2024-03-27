import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(test_labels, predictions):
    conf_matrix = confusion_matrix(test_labels, predictions)

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = np.unique(test_labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    return plt
