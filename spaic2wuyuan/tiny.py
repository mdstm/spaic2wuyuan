import spaic


class TinyModel(spaic.Network):
    def __init__(self):
        super().__init__(name='tiny')
        self.enc = spaic.Encoder(shape=(3, 4, 5), coding_method='poisson')

        self.neg0 = spaic.NeuronGroup(shape=(1, 3, 3), model='IF')
        self.neg1 = spaic.NeuronGroup(shape=(5,), model='LIF')
        self.neg2 = spaic.NeuronGroup(shape=(5,), model='CLIF')

        self.con0 = spaic.Connection(pre=self.enc, post=self.neg0, link_type='conv', w_mean=0.5, in_channels=3, kernel_size=(2, 3), padding=1, stride=2)
        self.con1 = spaic.Connection(pre=self.neg0, post=self.neg1, link_type='full', w_mean=0.5, syn_type=['flatten', 'basic'])
        self.con2 = spaic.Connection(pre=self.neg1, post=self.neg2, link_type='one_to_one', w_mean=0.5)

        self.dec0 = spaic.Decoder(dec_target=self.neg2, num=5, coding_method='spike_counts')
        self.dec1 = spaic.Decoder(dec_target=self.neg0, num=9, coding_method='spike_counts', coding_var_name='Isyn')

        self.mon0 = spaic.StateMonitor(target=self.neg2, var_name='O')
        self.mon1 = spaic.StateMonitor(target=self.neg0, var_name='V', index=((0, 0, 0, 0), (0, 1, 1, 2), (1, 0, 2, 1)))
        self.mon2 = spaic.StateMonitor(target=self.con0, var_name='weight', index=((0, 0), (0, 2), (0, 1), (0, 2)))
        self.mon3 = spaic.StateMonitor(target=self.con1, var_name='weight', index=((0, 1, 2), (2, 3, 7)))
        self.mon4 = spaic.StateMonitor(target=self.con2, var_name='weight', index=((0, 1, 2, 3, 4), (0, 1, 2, 3, 4)))

        b = spaic.Torch_Backend()
        b.dt = 1
        self.build(b)


class SmallModel(spaic.Network):
    def __init__(self):
        super().__init__(name='small')
        self.asm0 = spaic.Assembly()
        self.asm1 = spaic.Assembly()
        self.asm1.asm0 = spaic.Assembly()

        self.asm0.enc = spaic.Encoder(shape=(3, 4, 5), coding_method='poisson')

        self.asm1.neg = spaic.NeuronGroup(shape=(1, 3, 3), model='IF')
        self.asm1.asm0.neg = spaic.NeuronGroup(shape=(5,), model='LIF')
        self.neg = spaic.NeuronGroup(shape=(5,), model='CLIF')

        self.asm1.con = spaic.Connection(pre=self.asm0.enc, post=self.asm1.neg, link_type='conv', w_mean=0.5, in_channels=3, kernel_size=(2, 3), padding=1, stride=2)
        self.asm1.asm0.con = spaic.Connection(pre=self.asm1.neg, post=self.asm1.asm0.neg, link_type='full', w_mean=0.5, syn_type=['flatten', 'basic'])
        self.asm0.con = spaic.Connection(pre=self.asm1.asm0.neg, post=self.neg, link_type='one_to_one', w_mean=0.5)

        self.asm1.asm0.dec = spaic.Decoder(dec_target=self.neg, num=5, coding_method='spike_counts')
        self.asm1.dec = spaic.Decoder(dec_target=self.asm1.neg, num=9, coding_method='spike_counts', coding_var_name='Isyn')

        self.mon0 = spaic.StateMonitor(target=self.neg, var_name='O')
        self.mon1 = spaic.StateMonitor(target=self.asm1.neg, var_name='V', index=((0, 0, 0, 0), (0, 1, 1, 2), (1, 0, 2, 1)))
        self.mon2 = spaic.StateMonitor(target=self.asm1.con, var_name='weight', index=((0, 0), (0, 2), (0, 1), (0, 2)))
        self.mon3 = spaic.StateMonitor(target=self.asm1.asm0.con, var_name='weight', index=((0, 1, 2), (2, 3, 7)))
        self.mon4 = spaic.StateMonitor(target=self.asm0.con, var_name='weight', index=((0, 1, 2, 3, 4), (0, 1, 2, 3, 4)))

        b = spaic.Torch_Backend()
        b.dt = 1
        self.build(b)


class ActorNetSpiking(spaic.Network):
    def __init__(self, nornal_size=6, scan_size=360, action_num=2):
        super().__init__('ActorNetSpiking')
        self.nornal_size = nornal_size
        self.scan_size = scan_size
        self.state_num = 216
        self.v_th = 0.5

        self.input = spaic.Encoder(num=nornal_size+scan_size, coding_method='poisson', unit_conversion=10)

        self.layer1 = spaic.NeuronGroup(num=self.state_num, model='lif', v_th=self.v_th)
        self.layer2 = spaic.NeuronGroup(num=self.state_num, model='lif', v_th=self.v_th)
        self.layer3 = spaic.NeuronGroup(num=self.state_num , model='lif', v_th=self.v_th)

        self.layer5 = spaic.NeuronGroup(num=256 , model='lif', v_th=self.v_th)
        self.layer6 = spaic.NeuronGroup(num=256 , model='lif', v_th=self.v_th)
        self.layer7 = spaic.NeuronGroup(num=256 , model='lif', v_th=self.v_th)
        self.layer8 = spaic.NeuronGroup(num=action_num, model='lif', v_th=self.v_th)

        self.connection1 = spaic.Connection(pre=self.input, post=self.layer1, link_type='full')
        self.connection2 = spaic.Connection(pre=self.layer1, post=self.layer2, link_type='full')
        self.connection3 = spaic.Connection(pre=self.layer2, post=self.layer3, link_type='full')
        self.connection5 = spaic.Connection(pre=self.layer3, post=self.layer5, link_type='full')
        self.connection6 = spaic.Connection(pre=self.layer5, post=self.layer6, link_type='full')
        self.connection7 = spaic.Connection(pre=self.layer6, post=self.layer7, link_type='full')
        self.connection8 = spaic.Connection(pre=self.layer7, post=self.layer8, link_type='full')

        self.output = spaic.Decoder(num=action_num, dec_target=self.layer8, coding_method='spike_counts')

        self.mon_V8 = spaic.StateMonitor(self.layer8, 'V')

        self.mon_V1 = spaic.StateMonitor(self.layer1, 'V')
        self.mon_V2 = spaic.StateMonitor(self.layer2, 'V')
        self.mon_V3 = spaic.StateMonitor(self.layer3, 'V')
        self.mon_V5 = spaic.StateMonitor(self.layer5, 'V')
        self.mon_V6 = spaic.StateMonitor(self.layer6, 'V')
        self.mon_V7 = spaic.StateMonitor(self.layer7, 'V')

        self.mon_O1 = spaic.StateMonitor(self.layer1, 'O')
        self.mon_O2 = spaic.StateMonitor(self.layer2, 'O')
        self.mon_O3 = spaic.StateMonitor(self.layer3, 'O')

        self.mon_O5 = spaic.StateMonitor(self.layer5, 'O')
        self.mon_O6 = spaic.StateMonitor(self.layer6, 'O')
        self.mon_O7 = spaic.StateMonitor(self.layer7, 'O')
        self.mon_O8 = spaic.StateMonitor(self.layer8, 'O')

        self.mon_I1 =  spaic.StateMonitor(self.layer1, 'Isyn')
        self.mon_I2 =  spaic.StateMonitor(self.layer2, 'Isyn')

        b = spaic.Torch_Backend()
        b.dt = 1
        self.build(b)
