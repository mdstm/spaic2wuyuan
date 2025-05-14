import spaic


def traverse_spaic(net: spaic.Network) -> tuple[list, list, list, list]:
    '''深度遍历 spaic 网络，提取 Encoder, NeuronGroup, Decoder, Connection'''

    encoders = []
    neuron_groups = []
    decoders = []
    connections = []

    # dfs
    stack = [net]
    while stack:
        assem = stack.pop()
        for a in assem._groups.values():
            if isinstance(a, spaic.Encoder):
                encoders.append((a.id, a)) # 使用 id 避免名称重复
            elif isinstance(a, spaic.NeuronGroup):
                neuron_groups.append((a.id, a))
            elif isinstance(a, spaic.Decoder):
                decoders.append((a.id, a))
            elif a._groups:
                stack.append(a)
        for a in assem._connections.values():
            connections.append((a.id, a))

    return encoders, neuron_groups, decoders, connections
