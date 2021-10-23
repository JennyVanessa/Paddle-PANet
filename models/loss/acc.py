import paddle

EPS = 1e-6


def acc_single(a, b, mask):
    ind = mask == 1
    if paddle(ind) == 0:
        return 0
    correct = (a[ind] == b[ind]).float()
    acc = paddle.sum(correct) / correct.size(0)
    return acc


def acc(a, b, mask, reduce=True):
    batch_size = a.size(0)

    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    acc = a.new_zeros((batch_size, ), dtype="float32")
    for i in range(batch_size):
        acc[i] = acc_single(a[i], b[i], mask[i])

    if reduce:
        acc = paddle.mean(acc)
    return acc
