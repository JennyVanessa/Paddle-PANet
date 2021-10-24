import paddle


def init_weights(init_type='constant'):
    if init_type == 'constant':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant())
    if init_type == 'normal':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal())
    elif init_type == 'xavier':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
    elif init_type == 'kaiming':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
