import numpy as np
import spaic

from .extracter import vars


def get_value(a: spaic.BaseModule, var_name: str) -> np.ndarray:
    '''从后端中提取变量值，并转换为 numpy 类型'''
    backend = a._backend
    value = backend.get_varialble(var_name)
    value = backend.to_numpy(value)
    return value


def get_mon_info(a: spaic.StateMonitor, infos: dict) -> dict:
    '''获取监视器的所有信息，需要现有信息字典'''

    target = a.target
    target_id = target.id
    target_info = infos[target_id]
    target_type = target_info['type']
    if target_type not in {'NeuronGroup', 'ConnectionGroup'}:
        raise TypeError('Monitor 目标只能是 NeuronGroup 或 ConnectionGroup')

    # 默认信息
    param = {
        'sampling_period': float(a.dt),
    }
    info = {
        'target': target_id,
        'param': param,
    }

    # 下面开始设置观测的状态名称和位置
    var_name = a.var_name
    # 从长名称中提取大括号括住的短名称
    var = var_name[(i := var_name.find('{') + 1) : var_name.find('}', i)]
    index = a.index # 观测位置索引，可以是 'full' 或元组
    if var == 'O':
        info['type'] = 'SpikeMonitor'
        if index == 'full':
            param['position'] = np.ones(
                target_info['param']['shape'], dtype=bool,
            )
        else:
            # 先构造零数组，然后将索引位置设为真
            position = np.zeros(target_info['param']['shape'], dtype=bool)
            # 索引可能多一个批次维度，忽略
            position[index if len(index) == position.ndim else index[1:]] = True
            param['position'] = position

    elif target_type == 'NeuronGroup':
        info['type'] = 'StateMonitorNeuron'
        state_name, reshape = vars[target.model.__class__.__name__][var]
        param['state_name'] = state_name
        if index == 'full':
            param['position'] = np.ones_like(
                target_info['param']['initial_state_value'][state_name][0],
                dtype=bool,
            )
        else:
            # 先从后端获取原始形状，设置位置后再转换为 wuyuan 形状
            # 由于 spaic 多一个批次维度，为了节省内存，先去掉，等形变时再加回来
            position = np.zeros_like(get_value(target, var_name)[0], dtype=bool)
            position[index if len(index) == position.ndim else index[1:]] = True
            param['position'] = reshape(
                position[None, ...], target_info['param']['shape'],
            )

    else: # target_type == 'ConnectionGroup'
        info['type'] = 'StateMonitorSynapse'
        state_name, reshape = vars[target.__class__.__name__][var]
        param['state_name'] = state_name
        if index == 'full':
            param['position'] = np.ones_like(
                target_info['param']
                    ['initial_synapse_state_value'][state_name][0],
                dtype=bool,
            )
        else:
            position = np.zeros_like(get_value(target, var_name), dtype=bool)
            position[index] = True
            target_t_model_param = target_info['t_model_param']
            param['position'] = reshape(
                position,
                target_t_model_param['presynaptic_shape'],
                target_t_model_param['postsynaptic_shape'],
            )

    return info
