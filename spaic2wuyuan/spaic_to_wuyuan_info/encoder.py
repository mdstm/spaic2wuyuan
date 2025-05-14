import spaic

from .extracter import Extracter, update_info


def get_enc_info(a: spaic.Encoder) -> dict:
    '''获取编码器的所有信息'''

    info = {
        'type': 'Encoder',
        'param': {
            'shape': list(a.shape[1:]), # spaic 会给编码器形状增加一个维度
        },
        'model_type': 'unknown',
        'model_param': {
            'parameter': {},
        },
    }

    update_info(info, a.__class__.__name__, a)

    return info


class PoissonEncoding(Extracter):
    def get_info(a: spaic.Encoder) -> dict:
        return {
            'model_type': 'Poisson',
            'model_param': {
                'parameter': {
                    'spike_rate_per_unit': float(a.unit_conversion),
                    'time_step': float(a.dt),
                },
            },
        }
