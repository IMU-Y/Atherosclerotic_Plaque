import numpy as np

def confusion_matrix_statistics(sum_confusion_matrix):
    n_classes = sum_confusion_matrix.shape[0]
    metrics_result = []
    print('统计结果')
    print('-' * 20)
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(sum_confusion_matrix)
        # 对角线上是正确预测的
        TP = sum_confusion_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(sum_confusion_matrix[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(sum_confusion_matrix[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN

        # 统计指标
        smooth = 1e-3
        sensitivity = TP / (TP + FN + smooth)
        specificity = TN / (FP + TN + smooth)
        accuracy = (TP + TN) / (TP + FP + TN + FN + smooth)
        precision = TP / (TP + FP + smooth)
        recall = sensitivity
        f1_score = 2 * (precision * recall) / (precision + recall + smooth)
        # print('分类{}: TP:{}, TN:{}, FP:{}, FN:{}'.format(i+1, TP, TN, FP, FN))
        print('分类{}: sen:{:.2f}, spe:{:.2f}, acc:{:.2f}, precision:{:.2f}, recall:{:.2f}, f1_score:{:.2f}'.format(i+1, sensitivity, specificity, accuracy, precision, recall, f1_score))
        # metrics_result.append([TP / (TP + FP), TP / (TP + FN), TN / (TN + FP)])
    # return metrics_result
    print('-'*20)