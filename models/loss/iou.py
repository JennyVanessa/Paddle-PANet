import paddle

EPS = 1e-6


def iou_single(a, b, mask, n_class):
    valid = mask == 1
    a = a.numpy()
    b = b.numpy()
    valid = valid.numpy()
    a = a[valid]
    b = b[valid]
    a = paddle.to_tensor(a)
    b = paddle.to_tensor(b)
    miou = []
    for i in range(n_class):
        if a.size == 0:
            return 0

        inter = (a == i).logical_and((b == i)).cast('float32')
        union = (a == i).logical_or((b == i)).cast('float32')

        miou.append(paddle.sum(inter) / (paddle.sum(union) + EPS))

    miou = sum(miou) / len(miou)

    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = paddle.reshape(a, (batch_size, -1))
    b = paddle.reshape(b, (batch_size, -1))
    mask = paddle.reshape(mask, (batch_size, -1))

    iou = paddle.zeros((batch_size,), dtype='float32')
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = paddle.mean(iou)
    return iou
