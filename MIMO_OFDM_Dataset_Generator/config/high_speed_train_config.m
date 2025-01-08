function config = high_speed_train_config()
    % 高铁场景的配置文件
    config.scenario = 'High-Speed Train';  % 场景名称
    config.channelModel = 'TDL-C';         % 信道模型
    config.numSubcarriers = 64;            % OFDM子载波数量
    config.numTxAntennas = 4;              % 发射天线数量
    config.numRxAntennas = 4;              % 接收天线数量
    config.samplingRate = 30.72e6;         % 采样率 (5G标准)
    config.carrierFrequency = 3.5e9;       % 载波频率 (5G中频段)
    config.maximumDopplerShift = 500;      % 最大多普勒频移 (根据速度计算)
    config.numSamples = 10000;             % 数据集大小
    config.savePath = 'datasets/high_speed_train.mat';  % 保存路径
end
