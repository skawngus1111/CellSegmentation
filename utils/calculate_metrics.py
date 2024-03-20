import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

class CellSegmentation_Metrics_Calculator(object):
    def __init__(self, metric_list):
        super(CellSegmentation_Metrics_Calculator, self).__init__()

        self.metric_list = metric_list
        self.total_metrics_dict = dict()
        for metric in self.metric_list:
            self.total_metrics_dict[metric] = list()

        self.smooth = 1e-5

    def get_metrics_dict(self, y_pred, y_true):
        y_true = y_true.squeeze().detach().cpu().numpy()
        y_pred = (y_pred.squeeze().detach().cpu().numpy() >= 0.5).astype(np.int_)

        y_true = np.asarray(y_true, np.float32)
        y_true /= (y_true.max() + self.smooth)
        y_true[y_true > 0.5] = 1; y_true[y_true != 1] = 0

        metric_dict = dict()

        for metric in self.metric_list:
            metric_dict[metric] = 0
            result = self.get_metrics(metric, y_pred, y_true)
            if np.isnan(result): result = 1e-6
            metric_dict[metric] = result

        return metric_dict

    def get_metrics(self, metric, y_pred, y_true):
        if metric == 'DSC': return self.calculate_DSC(y_pred, y_true)
        elif metric == 'IoU': return self.calculate_IoU(y_pred, y_true)
        elif metric == 'Precision': return self.calculate_Precision(y_pred, y_true)
        elif metric == 'Recall': return self.calculate_Recall(y_pred, y_true)
        elif metric == 'Specificity': return self.calculate_Specificity(y_pred, y_true)


    def calculate_DSC(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        intersection = np.sum(y_true * y_pred)
        return (2. * intersection + self.smooth) / (np.sum(y_true) + np.sum(y_pred) + self.smooth)

    def calculate_IoU(self, y_pred, y_true):
        y_pred_f = y_pred > 0.5
        y_true_f = y_true > 0.5

        intersection_f = (y_pred_f & y_true_f).sum()
        union_f = (y_pred_f | y_true_f).sum()

        iou_f = (intersection_f + self.smooth) / (union_f + self.smooth)

        return iou_f

    def calculate_Precision(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        return precision_score(y_true, y_pred)

    def calculate_Recall(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        return recall_score(y_true, y_pred)

    def calculate_Specificity(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        cm = list(confusion_matrix(y_true, y_pred).ravel())

        if len(cm) == 1: cm += [0, 0, 0]

        tn, fp, fn, tp = cm
        specificity = tn / (tn+fp)

        return specificity
