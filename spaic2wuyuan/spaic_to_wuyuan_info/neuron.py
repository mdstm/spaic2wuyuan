import numpy as np
from numpy import int16
import spaic

from .extracter import Extracter, update_info


def get_neg_value(a: spaic.NeuronGroup, var: str) -> np.ndarray:
    '''从后端中提取神经元组变量值，并转换为 numpy 类型'''
    backend = a._backend
    value = backend.get_varialble(a.get_labeled_name(var))
    value = backend.to_numpy(value)
    return value


def get_neg_info(a: spaic.NeuronGroup) -> dict:
    '''获取神经元组的所有信息'''

    info = {
        'type': 'NeuronGroup',
        'param': {
            'shape': list(a.shape),
            'initial_state_value': {},
        },
        'model_type': 'unknown',
        'model_param': {
            'parameter': {},
            'initial_state': {},
        },
    }

    update_info(info, a.model.__class__.__name__, a)

    return info


def reshape(arr: np.ndarray, *args) -> np.ndarray:
    '''神经元组变量默认的形变函数。去除 spaic 后端的批数维度。'''
    return arr[0]


class LIFModel(Extracter):
    var_dict = {
        'Isyn': ('weight_sum', reshape),
        'V': ('voltage', reshape),
    }

    def get_info(a: spaic.NeuronGroup) -> dict:
        # spaic 的神经元组虽然有自己的 dt，但将 tau 值放入后端会使用后端的 dt
        dt = float(a._backend.dt)
        return {
            'model_type': 'LIF',
            'model_param': {
                'parameter': {
                    # tauM = RC, 这里直接将电容设为 tauM, 电阻设为 1
                    'capacitance': float(a.model._tau_variables['tauM']),
                    'resistance': 1.0,
                    'time_step': dt,
                    'voltage_rest': int16(0), # spaic 没有静息，设为 0
                    # 电位变量需要从后端获取量化值
                    'threshold': int16(get_neg_value(a, 'Vth')),
                    'voltage_reset_value': int16(get_neg_value(a, 'Vreset')),
                    'refractory_period': dt, # spaic 没有不应期，设为 dt
                    'voltage_initial': int16(0),
                    'c_weight_sum': 1.0,
                },
            },
            'param': {
                'initial_state_value': {
                    # 从后端获取状态值并做转换
                    state_name: (
                        reshape(get_neg_value(a, var)).astype(int16),
                        False, # 神经元组的状态值都不是常量
                    ) for var, (state_name, reshape) in
                        LIFModel.var_dict.items()
                },
            },
        }


class IFModel(Extracter):
    var_dict = LIFModel.var_dict

    def get_info(a: spaic.NeuronGroup) -> dict:
        dt = float(a._backend.dt)
        return {
            'model_type': 'LIF',
            'model_param': {
                'parameter': {
                    'capacitance': np.inf, # 电容无穷代表这是 IF 模型
                    'resistance': 1.0, # 总之使 exp(-dt/RC) == 1
                    'time_step': dt,
                    # 借用静息电位存储恒定增量
                    'voltage_rest': -int16(get_neg_value(a, 'ConstantDecay')),
                    'threshold': int16(get_neg_value(a, 'Vth')),
                    'voltage_reset_value': int16(0), # spaic 没有重置，设为 0
                    'refractory_period': dt,
                    'voltage_initial': int16(0),
                    'c_weight_sum': 1.0,
                },
            },
            'param': {
                'initial_state_value': {
                    state_name: (
                        reshape(get_neg_value(a, var)).astype(int16),
                        False,
                    ) for var, (state_name, reshape) in IFModel.var_dict.items()
                },
            },
        }


class CLIFModel(Extracter):
    var_dict = {
        'Isyn': ('weight_sum', reshape),
        'M': ('voltage_m', reshape),
        'S': ('voltage_s', reshape),
        'E': ('voltage_e', reshape),
    }

    def get_info(a: spaic.NeuronGroup) -> dict:
        tau_variables = a.model._tau_variables
        return {
            'model_type': 'CLIF',
            'model_param': {
                'parameter': {
                    'tau_m': float(tau_variables['tauP']),
                    'tau_s': float(tau_variables['tauQ']),
                    'tau_e': float(tau_variables['tauM']),
                    'time_step': float(a._backend.dt),
                    'threshold': int16(get_neg_value(a, 'Vth')),
                    'c_weight_sum_m': 1.0,
                    'c_weight_sum_s': 1.0,
                },
            },
            'param': {
                'initial_state_value': {
                    state_name: (
                        reshape(get_neg_value(a, var)).astype(int16),
                        False,
                    ) for var, (state_name, reshape) in
                        CLIFModel.var_dict.items()
                },
            },
        }
