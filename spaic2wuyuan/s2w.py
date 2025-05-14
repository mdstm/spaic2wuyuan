import spaic

from wuyuan.snn_model import (
    Network,
    Encoder,
    NeuronGroup,
    ConnectionGroup,
    SpikeDecoder,
    StateDecoderNeuron,
    SpikeMonitor,
    StateMonitorNeuron,
    StateMonitorSynapse,
    encoder_models,
    neuron_models,
    topology_models,
    synapse_models,
    decoder_models,
)

from spaic_to_wuyuan_info import get_infos


def spaic2wuyuan(net_sp: spaic.Network) -> Network:
    '''把 spaic 表示的 SNN 转换为物源'''

    net_param, infos = get_infos(net_sp)

    net = Network()
    net.set_time_step(net_param['time_step'])
    net.set_time_window(net_param['time_window'])

    # 存储已构造的编码器、神经元组和连接组，用于构造连接组、解码器和监视器
    elements: dict[str] = {}

    for name, info in infos.items():
        match info['type']:
            case 'Encoder':
                model = net.encoder_model_get('0', # 随便填的模型 id
                    getattr(encoder_models, info['model_type']) # 获取模型类
                        (**info['model_param']), # 调用构造函数
                )
                a = net.encoder_add(Encoder(model, **info['param']), name)
                elements[name] = a

            case 'NeuronGroup':
                model = net.neuron_model_get('0',
                    getattr(neuron_models, info['model_type'])
                        (**info['model_param']),
                )
                a = net.neuron_group_add(NeuronGroup(
                    model, **info['param'],
                ), name)
                elements[name] = a

            case 'ConnectionGroup':
                t_model = net.topology_model_get('0',
                    getattr(topology_models, info['t_model_type'])
                        (**info['t_model_param']),
                )
                s_model = net.synapse_model_get('0',
                    getattr(synapse_models, info['s_model_type'])
                        (**info['s_model_param']),
                )
                a = net.connection_group_add(ConnectionGroup(
                    presynaptic_node=elements[info['pre']],
                    postsynaptic_node=elements[info['post']],
                    topology_model=t_model,
                    synapse_model=s_model,
                    **info['param'],
                ), name)
                elements[name] = a

            case 'SpikeDecoder':
                model = net.decoder_model_get('0',
                    getattr(decoder_models, info['model_type'])
                        (**info['model_param']),
                )
                net.spike_decoder_add(SpikeDecoder(
                    decoder_model=model,
                    target_element=elements[info['target']],
                    **info['param']
                ), name)
            case 'StateDecoderNeuron':
                model = net.decoder_model_get('0',
                    getattr(decoder_models, info['model_type'])
                        (**info['model_param']),
                )
                net.state_decoder_neuron_add(StateDecoderNeuron(
                    decoder_model=model,
                    target_element=elements[info['target']],
                    **info['param']
                ), name)
            # 没有 StateDecoderSynapse

            case 'SpikeMonitor':
                net.spike_monitor_add(SpikeMonitor(
                    elements[info['target']], **info['param'],
                ), name)
            case 'StateMonitorNeuron':
                net.state_monitor_neuron_add(StateMonitorNeuron(
                    elements[info['target']], **info['param'],
                ), name)
            case 'StateMonitorSynapse':
                net.state_monitor_synapse_add(StateMonitorSynapse(
                    elements[info['target']], **info['param'],
                ), name)

    return net


# 测试
def main():
    from quantize import quantize_net
    from test import test
    from tiny import TinyModel, SmallModel

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
