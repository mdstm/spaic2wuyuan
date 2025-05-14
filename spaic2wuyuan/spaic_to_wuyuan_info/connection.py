import numpy as np
from numpy import int16
import spaic

from .extracter import Extracter, update_info


def get_con_value(a: spaic.Connection, var: str) -> np.ndarray:
    '''从后端中提取连接变量值，并转换为 numpy 类型'''
    backend = a._backend
    value = backend.get_varialble(a.get_link_name(a.pre, a.post, var))
    value = backend.to_numpy(value)
    return value


def get_con_info(a: spaic.Connection, infos: dict) -> dict:
    '''获取连接的所有信息，需要现有信息字典'''

    pre_id, post_id = a.pre.id, a.post.id
    pre_shape = infos[pre_id]['param']['shape']
    post_shape = infos[post_id]['param']['shape']
    info = {
        'type': 'ConnectionGroup',
        'pre': pre_id,
        'post': post_id,
        'param': {
            'initial_synapse_state_value': {},
            'delay': float(a.max_delay),
        },
        't_model_type': 'unknown',
        't_model_param': {
            'presynaptic_shape': pre_shape,
            'postsynaptic_shape': post_shape,
            'parameter': {},
        },
        's_model_type': 'Delta', # spaic 没有突触模型，设为 Delta
        's_model_param': {
            'parameter': {'bit_width_weight': 16},
            'initial_state': {'weight': int16(1)},
        },
    }

    update_info(info, a.__class__.__name__, a, pre_shape, post_shape)

    return info


class FullConnection(Extracter):
    def reshape(arr: np.ndarray,
                pre_shape: list[int], post_shape: list[int]) -> np.ndarray:
        '''spaic: (post_num, pre_num) -> wuyuan: pre_shape + post_shape'''
        return arr.T.reshape(pre_shape + post_shape)

    var_dict = {
        'weight': ('weight', reshape),
    }

    def get_info(a: spaic.Connection,
                 pre_shape: list[int], post_shape: list[int]) -> dict:
        return {
            't_model_type': 'FullyConnected',
            'param': {
                'initial_synapse_state_value': {
                    # 从后端获取状态值并做转换
                    state_name: (
                        reshape(get_con_value(a, var), pre_shape, post_shape)
                            .astype(int16),
                        True, # 连接组的状态值都是常量
                    ) for var, (state_name, reshape) in
                        FullConnection.var_dict.items()
                },
            },
        }


class one_to_one_mask(Extracter):
    def reshape(arr: np.ndarray, *args) -> np.ndarray:
        '''spaic 把权重放在对角线上 -> wuyuan: (pre_num,)'''
        return arr.diagonal()

    var_dict = {
        'weight': ('weight', reshape),
    }

    def get_info(a: spaic.Connection,
                 pre_shape: list[int], post_shape: list[int]) -> dict:
        return {
            't_model_type': 'OneToOne',
            'param': {
                'initial_synapse_state_value': {
                    state_name: (
                        reshape(get_con_value(a, var)).astype(int16),
                        True,
                    ) for var, (state_name, reshape) in
                        one_to_one_mask.var_dict.items()
                },
            },
        }


class conv_connect(Extracter):
    def reshape(arr: np.ndarray, *args) -> np.ndarray:
        '''spaic 和 wuyuan 权重形状一样，都是 (Cout, Cin, H, W)'''
        return arr

    var_dict = {
        'weight': ('weight', reshape),
    }

    def get_info(a: spaic.Connection,
                 pre_shape: list[int], post_shape: list[int]) -> dict:
        return {
            't_model_type': 'Convolution2D',
            't_model_param': {
                'parameter': {
                    'kernel_size': a.kernel_size,
                    'padding': a.padding,
                    'stride': a.stride,
                    'dilation': (1, 1), # spaic 没有用到膨胀系数
                },
            },
            'param': {
                'initial_synapse_state_value': {
                    state_name: (
                        reshape(get_con_value(a, var)).astype(int16),
                        True,
                    ) for var, (state_name, reshape) in
                        conv_connect.var_dict.items()
                },
            },
        }
