import numpy as np
import spaic

from traverse_spaic import traverse_spaic


def get_value(b: spaic.Torch_Backend, var_name: str) -> np.ndarray:
    '''从后端提取数值，并转换为 numpy 类型'''
    return b.get_varialble(var_name).numpy(force=True)


def set_value(b: spaic.Torch_Backend, var_name: str, value: np.ndarray):
    '''设置后端数值'''
    b.set_variable_value(var_name, value, var_name in b._parameters_dict)


def quantize(b: spaic.Torch_Backend, var_name: str,
        xmin: float, xmax: float, ymin: float, ymax: float):
    '''将后端数据从 [xmin, xmax] 量化到 [ymin, ymax]'''
    value = get_value(b, var_name)
    value.clip(xmin, xmax, out=value)
    dy = ymax - ymin
    value -= ((ymax / dy) * xmin - (ymin / dy) * xmax)
    value *= (dy / (xmax - xmin))
    value.round(out=value)
    set_value(b, var_name, value)


def restore(b: spaic.Torch_Backend, var_name: str,
        xmin: float, xmax: float, ymin: float, ymax: float):
    '''将后端数据从 [ymin, ymax] 还原到 [xmin, xmax]'''
    value = get_value(b, var_name)
    value.clip(ymin, ymax, out=value)
    dy = ymax - ymin
    value *= ((xmax - xmin) / dy)
    value += ((ymax / dy) * xmin - (ymin / dy) * xmax)
    set_value(b, var_name, value)


var_neg = {
    'IFModel': ('ConstantDecay', 'Vth', 'Isyn', 'V'),
    'LIFModel': ('Vth', 'Vreset', 'Isyn', 'V'),
    'CLIFModel': ('Vth', 'Isyn', 'M', 'S', 'E'),
}

var_con = {
    'FullConnection': ('weight',),
    'one_to_one_mask': ('weight',),
    'conv_connect': ('weight',),
}


def quantize_neg(a: spaic.NeuronGroup,
        xmin=0., xmax=1., ymin=0., ymax=32767.):
    '''默认量化神经元组'''
    for var in var_neg[a.model.__class__.__name__]:
        quantize(a._backend, a.get_labeled_name(var),
            xmin, xmax, ymin, ymax)


def quantize_con(a: spaic.Connection,
        xmin=-1., xmax=1., ymin=-32768., ymax=32767.):
    '''默认量化连接'''
    for var in var_con[a.__class__.__name__]:
        quantize(a._backend, a.get_link_name(a.pre, a.post, var),
            xmin, xmax, ymin, ymax)


def restore_neg(a: spaic.NeuronGroup,
        xmin=0., xmax=1., ymin=0., ymax=32767.):
    '''默认还原神经元组'''
    for var in var_neg[a.model.__class__.__name__]:
        restore(a._backend, a.get_labeled_name(var),
            xmin, xmax, ymin, ymax)


def restore_con(a: spaic.Connection,
        xmin=-1., xmax=1., ymin=-32768., ymax=32767.):
    '''默认还原连接'''
    for var in var_con[a.__class__.__name__]:
        restore(a._backend, a.get_link_name(a.pre, a.post, var),
            xmin, xmax, ymin, ymax)


def quantize_net_neg(net: spaic.Network):
    '''默认量化所有神经元组'''
    _, neuron_groups, _, _ = traverse_spaic(net)
    for _, a in neuron_groups:
        quantize_neg(a)


def quantize_net_con(net: spaic.Network):
    '''默认量化所有连接'''
    _, _, _, connections = traverse_spaic(net)
    for _, a in connections:
        quantize_con(a)


def restore_net_neg(net: spaic.Network):
    '''默认还原所有神经元组'''
    _, neuron_groups, _, _ = traverse_spaic(net)
    for _, a in neuron_groups:
        restore_neg(a)


def restore_net_con(net: spaic.Network):
    '''默认还原所有连接'''
    _, _, _, connections = traverse_spaic(net)
    for _, a in connections:
        restore_con(a)


def quantize_net(net: spaic.Network):
    '''默认量化网络'''
    _, neuron_groups, _, connections = traverse_spaic(net)
    for _, a in neuron_groups:
        quantize_neg(a)
    for _, a in connections:
        quantize_con(a)


def restore_net(net: spaic.Network):
    '''默认还原网络'''
    _, neuron_groups, _, connections = traverse_spaic(net)
    for _, a in neuron_groups:
        restore_neg(a)
    for _, a in connections:
        restore_con(a)
