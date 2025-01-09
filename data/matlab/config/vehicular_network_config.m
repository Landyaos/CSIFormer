function config = vehicular_network_config()
    % 车联网场景的配置文件
    config.scenario = 'Vehicular Network'; % 场景名称
    config.channelModel = 'TDL-B';         % 信道模型
    config.numSubcarriers = 128;           % OFDM子载波数量
    config.numTxAntennas = 2;              % 发射天线数量
    config.numRxAntennas = 2;              % 接收天线数量
    config.samplingRate = 15.36e6;         % 采样率 (5G低频段)
    config.carrierFrequency = 2.1e9;       % 载波频率
    config.maximumDopplerShift = 300;      % 最大多普勒频移 (相对低速)
    config.numSamples = 20000;             % 数据集大小
    config.savePath = 'datasets/vehicular_network.mat'; % 保存路径
end
