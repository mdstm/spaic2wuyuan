import spaic

from wuyuan.snn_model.network import Network
from wuyuan.snn_model.encoder import Encoder
from wuyuan.snn_model.neuron_group import NeuronGroup
from wuyuan.snn_model.connection_group import ConnectionGroup
from wuyuan.snn_model.spike_decoder import SpikeDecoder
from wuyuan.snn_model.state_decoder_neuron import StateDecoderNeuron
from wuyuan.snn_model.spike_monitor import SpikeMonitor
from wuyuan.snn_model.state_monitor_neuron import StateMonitorNeuron
from wuyuan.snn_model.state_monitor_synapse import StateMonitorSynapse

from wuyuan.snn_model.encoder_models.poisson import Poisson
from wuyuan.snn_model.neuron_models.lif import LIF
from wuyuan.snn_model.neuron_models.clif import CLIF
from wuyuan.snn_model.topology_models.fully_connected import FullyConnected
from wuyuan.snn_model.topology_models.one_to_one import OneToOne
from wuyuan.snn_model.topology_models.convolution_2d import Convolution2D
from wuyuan.snn_model.synapse_models.Delta import Delta
from wuyuan.snn_model.decoder_models.spike_count import SpikeCount

from .spaic_to_wuyuan_info import get_infos


ENCODER_MODEL_CLASS = {
    'Poisson': Poisson,
}
NEURON_MODEL_CLASS = {
    'LIF': LIF,
    'CLIF': CLIF,
}
TOPOLOGY_MODEL_CLASS = {
    'FullyConnected': FullyConnected,
    'OneToOne': OneToOne,
    'Convolution2D': Convolution2D,
}
SYNAPSE_MODEL_CLASS = {
    'Delta': Delta,
}
DECODER_MODEL_CLASS = {
    'SpikeCount': SpikeCount,
}


def spaic2wuyuan(net_sp: spaic.Network) -> Network:
    '''把 spaic 网络转换为物源，spaic 网络应该被 build 过'''

    # 网络基本参数和各组件的信息
    net_param, infos = get_infos(net_sp)

    net = Network()
    net.set_time_step(net_param['time_step'])
    net.set_time_window(net_param['time_window'])

    # 存储已构造的编码器、神经元组和连接组，用于构造连接组、解码器和监视器
    elements: dict[str] = {}

    # 根据组件的类型，给物源网络添加相应的组件
    for name, info in infos.items():
        match info['type']:
            case 'Encoder':
                ModelClass = ENCODER_MODEL_CLASS[info['model_type']]
                model = net.encoder_model_get(
                    '0', # 随便填的模型 id
                    ModelClass(**info['model_param']),
                )
                a = net.encoder_add(
                    Encoder(encoder_model=model, **info['param']),
                    name,
                )
                elements[name] = a

            case 'NeuronGroup':
                ModelClass = NEURON_MODEL_CLASS[info['model_type']]
                model = net.neuron_model_get(
                    '0',
                    ModelClass(**info['model_param']),
                )
                a = net.neuron_group_add(
                    NeuronGroup(neuron_model=model, **info['param']),
                    name,
                )
                elements[name] = a

            case 'ConnectionGroup':
                TModelClass = TOPOLOGY_MODEL_CLASS[info['t_model_type']]
                t_model = net.topology_model_get(
                    '0',
                    TModelClass(**info['t_model_param']),
                )
                SModelClass = SYNAPSE_MODEL_CLASS[info['s_model_type']]
                s_model = net.synapse_model_get(
                    '0',
                    SModelClass(**info['s_model_param']),
                )
                a = net.connection_group_add(
                    ConnectionGroup(
                        presynaptic_node=elements[info['pre']],
                        postsynaptic_node=elements[info['post']],
                        topology_model=t_model,
                        synapse_model=s_model,
                        **info['param'],
                    ),
                    name,
                )
                elements[name] = a

            case 'SpikeDecoder':
                ModelClass = DECODER_MODEL_CLASS[info['model_type']]
                model = net.decoder_model_get(
                    '0',
                    ModelClass(**info['model_param']),
                )
                net.spike_decoder_add(
                    SpikeDecoder(
                        decoder_model=model,
                        target_element=elements[info['target']],
                        **info['param'],
                    ),
                    name,
                )
            case 'StateDecoderNeuron':
                ModelClass = DECODER_MODEL_CLASS[info['model_type']]
                model = net.decoder_model_get(
                    '0',
                    ModelClass(**info['model_param']),
                )
                net.state_decoder_neuron_add(
                    StateDecoderNeuron(
                        decoder_model=model,
                        target_element=elements[info['target']],
                        **info['param'],
                    ),
                    name,
                )
            # 由于 spaic 特性，没有 StateDecoderSynapse

            case 'SpikeMonitor':
                net.spike_monitor_add(
                    SpikeMonitor(
                        target_element=elements[info['target']],
                        **info['param'],
                    ),
                    name,
                )
            case 'StateMonitorNeuron':
                net.state_monitor_neuron_add(
                    StateMonitorNeuron(
                        target_element=elements[info['target']],
                        **info['param'],
                    ),
                    name,
                )
            case 'StateMonitorSynapse':
                net.state_monitor_synapse_add(
                    StateMonitorSynapse(
                        elements[info['target']],
                        **info['param'],
                    ),
                    name,
                )

    return net


# 测试
def main():
    from .quantize import quantize_net
    from .test import test
    from .tiny import TinyModel, SmallModel

    @test('构造 spaic')
    def net_sp():
        return SmallModel()

    @test('量化')
    def _():
        quantize_net(net_sp)

    @test('转换 spaic -> wuyuan')
    def net():
        return spaic2wuyuan(net_sp)

    return net_sp, net


if __name__ == '__main__':
    main()
