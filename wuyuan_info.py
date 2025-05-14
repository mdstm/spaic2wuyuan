'''构造物源 SNN 模型需要的信息'''

import numpy as np


{
    '{enc_id}': {
        'type': 'Encoder',
        'param': {
            'shape': list[int],
        },
        'model_type': str,
        'model_param': {
            'parameter': dict,
        },
    },
    '{neg_id}': {
        'type': 'NeuronGroup',
        'param': {
            'shape': list[int],
            'initial_state_value': dict[str, tuple[np.ndarray, bool]],
        },
        'model_type': str,
        'model_param': {
            'parameter': dict,
            'initial_state': dict,
        },
    },
    '{con_id}': {
        'type': 'ConnectionGroup',
        'pre': str,
        'post': str,
        'param': {
            'initial_synapse_state_value': dict[str, tuple[np.ndarray, bool]],
            'delay': float,
        },
        't_model_type': str,
        't_model_param': {
            'presynaptic_shape': list[int],
            'postsynaptic_shape': list[int],
            'parameter': dict,
        },
        's_model_type': str,
        's_model_param': {
            'parameter': dict,
            'initial_state': dict,
        },
    },
    '{spdec_id}': {
        'type': 'SpikeDecoder',
        'target': str,
        'param': {
            'sampling_period': float,
            'position': np.ndarray,
        },
        'model_type': str,
        'model_param': {
            'parameter': dict,
        },
    },
    '{stdec_id}': {
        'type': 'StateDecoderNeuron | StateDecoderSynapse',
        'target': str,
        'param': {
            'sampling_period': float,
            'position': np.ndarray,
            'state_name': str,
        },
        'model_type': str,
        'model_param': {
            'parameter': dict,
        },
    },
    '{spmon_id}': {
        'type': 'SpikeMonitor',
        'target': str,
        'param': {
            'sampling_period': float,
            'position': np.ndarray,
        },
    },
    '{stmon_id}': {
        'type': 'StateMonitorNeuron | StateMonitorSynapse',
        'target': str,
        'param': {
            'sampling_period': float,
            'position': np.ndarray,
            'state_name': str,
        },
    },
}
