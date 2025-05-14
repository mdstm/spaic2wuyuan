import numpy as np
import spaic

from .extracter import Extracter, update_info, vars


def get_dec_info(a: spaic.Decoder, infos: dict) -> dict:
    '''获取解码器的所有信息，需要现有信息字典'''

    target = a.dec_target
    target_id = target.id
    target_info = infos[target_id]
    if target_info['type'] != 'NeuronGroup':
        raise TypeError('Decoder 目标只能是 NeuronGroup')

    # 默认信息
    info = {
        'target': target_id,
        'param': (param := {
            'sampling_period': float(a.dt),
        }),
        'model_type': 'unknown',
        'model_param': {
            'parameter': {},
        },
    }

    # 设置观测的状态名称和位置
    var = a.coding_var_name
    if var == 'O':
        info['type'] = 'SpikeDecoder'
        param['position'] = np.ones(target_info['param']['shape'], dtype=bool)
    else:
        info['type'] = 'StateDecoderNeuron'
        state_name, _ = vars[target.model.__class__.__name__][var]
        param['state_name'] = state_name
        param['position'] = np.ones_like(
            target_info['param']['initial_state_value'][state_name][0],
            dtype=bool,
        )

    update_info(info, a.__class__.__name__, a)

    return info


class Spike_Counts(Extracter):
    def get_info(a: spaic.Decoder) -> dict:
        return {
            'model_type': 'SpikeCount',
        }
