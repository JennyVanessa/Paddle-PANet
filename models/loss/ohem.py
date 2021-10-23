import paddle
import numpy as np

def ohem_single(score, gt_text, training_mask):
    pos = paddle.cast(gt_text > 0.5, dtype='int32')
    neglect = paddle.cast(paddle.logical_and((gt_text > 0.5), (training_mask <= 0.5)), dtype='int32')
    pos_num = paddle.sum(pos) - paddle.sum(neglect)
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = paddle.reshape(selected_mask, [1, selected_mask.shape[0], selected_mask.shape[1]])
        selected_mask = paddle.cast(selected_mask, 'float32')
        return selected_mask
    neg = paddle.cast(gt_text <= 0.5, dtype='int32')
    neg_num = paddle.sum(neg)
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = paddle.reshape(selected_mask, [1, selected_mask.shape[0], selected_mask.shape[1]])
        selected_mask = paddle.cast(selected_mask, 'float32')
        return selected_mask
    # neg_score = score[gt_text <= 0.5]
    neg_score = paddle.to_tensor(score.numpy()[gt_text.numpy() <= 0.5])
    neg_score_sorted = paddle.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]
    selected_mask = (score >= threshold).logical_or(gt_text > 0.5).logical_and(training_mask > 0.5)
    selected_mask = paddle.reshape(selected_mask, [1, selected_mask.shape[0], selected_mask.shape[1]])
    selected_mask = paddle.cast(selected_mask, 'float32')
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    gt_texts = paddle.cast(gt_texts, dtype='int32')
    training_masks = paddle.cast(training_masks, dtype='int32')
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = paddle.concat(selected_masks, 0)
    selected_masks = paddle.cast(selected_masks, 'float32')

    return selected_masks

#
# input = paddle.to_tensor([[[0.1, -0.2, 0.3], [0.4, -0.3, -0.5]]])
# target = paddle.to_tensor([[[1, 1, 0], [0, 1, 1]]])
# mask = paddle.to_tensor([[[1, 1, 1], [0, 0, 0]]])
#
#
# print(ohem_batch(input, target, mask))