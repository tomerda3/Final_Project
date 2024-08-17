import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

GENERAL_LABELS = [0, 1, 2, 3]
CONFUSION_MATRIX_SAVE_DIRECTORY = './confusion_matrix_images'


class ConfusionMatrixGenerator:

    def __init__(self):
        self.labels = GENERAL_LABELS
        self.save_dir = CONFUSION_MATRIX_SAVE_DIRECTORY

    def build_confusion_matrix_plot(self, test_labels: List[str], predictions: List[str], model_name: str, data_name: str):
        cm = confusion_matrix(test_labels, predictions, labels=self.labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f"{self.save_dir}/{data_name}/{model_name}")
        plt.close()
