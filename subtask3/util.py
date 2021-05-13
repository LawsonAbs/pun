import torch as t
# 这个好像是从苏剑林那里获取的一个多标签分类方法
# 如果有的y_true 只有一个1，是因为只留了前n个definition，所以截断了
def multilabel_crossentropy(y_pred,y_true):
    # 下面三行代码主要是为了找出pred中的正负样例
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = t.zeros_like(y_pred[..., :1])
    y_pred_neg = t.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = t.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = t.logsumexp(y_pred_neg, dim=-1)
    pos_loss = t.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss
    return t.sum(loss)
    # return neg_loss + pos_loss


if __name__ == "__main__":
    y_pred = t.tensor([[ 116.9767,   8.2804,   2.9881, -80.4905,  15.1279,  50.4803,  37.6280, 41.1969, -24.1459,  62.4197],
                        [ 116.9767,   8.2804,   2.9881, -80.4905,  15.1279,  50.4803,  37.6280, 41.1969, -24.1459,  62.4197]])
    y_true = t.tensor([[0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 0., 0., 0., 0.]])
    loss = multilabel_crossentropy(y_pred,y_true)
    print(loss)
