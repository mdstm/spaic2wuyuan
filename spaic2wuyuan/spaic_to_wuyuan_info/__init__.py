import spaic

from traverse_spaic import traverse_spaic

from .encoder import get_enc_info
from .neuron import get_neg_info
from .connection import get_con_info
from .decoder import get_dec_info
from .monitor import get_mon_info


def get_infos(net: spaic.Network) -> tuple[dict, dict]:
    '''将 spaic 转换为物源信息，返回 Network 属性和元素信息'''

    encoders, neuron_groups, decoders, connections = traverse_spaic(net)

    # 开始获取元素信息
    infos = {}

    # 解析编码器
    for id, a in encoders:
        infos[id] = get_enc_info(a)

    # 解析神经元组
    for id, a in neuron_groups:
        infos[id] = get_neg_info(a)

    # 解析连接
    for id, a in connections:
        infos[id] = get_con_info(a, infos)

    # 解析解码器
    for id, a in decoders:
        infos[id] = get_dec_info(a, infos)

    # 解析监视器
    for name, a in net._monitors.items():
        infos[name] = get_mon_info(a, infos)

    # 获取 Network 属性
    backend = net._backend
    net_param = {
        'time_step': backend.dt,
        'time_window': backend.runtime,
    }

    return net_param, infos
