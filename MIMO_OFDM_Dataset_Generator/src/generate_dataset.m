function generate_dataset(configFile)
    % 主函数：根据配置文件生成数据集
    % 输入参数:
    % - configFile: 配置文件路径，例如 'config/high_speed_train_config.m'
    
    % 加载配置
    configFunc = str2func(configFile);
    config = configFunc();
    
    % 显示场景信息
    fprintf('Generating dataset for %s scenario...\n', config.scenario);
    
    % 创建信道对象
    tdlChannel = nrTDLChannel;
    tdlChannel.DelayProfile = config.channelModel;
    tdlChannel.NumTransmitAntennas = config.numTxAntennas;
    tdlChannel.NumReceiveAntennas = config.numRxAntennas;
    tdlChannel.MaximumDopplerShift = config.maximumDopplerShift;
    tdlChannel.SampleRate = config.samplingRate;

    % 数据集存储初始化
    txSignals = cell(config.numSamples, 1);
    rxSignals = cell(config.numSamples, 1);
    channelMatrices = cell(config.numSamples, 1);

    % 数据生成循环
    for i = 1:config.numSamples
        % 生成随机OFDM信号
        txSignal = generate_ofdm_signal(config.numSubcarriers, config.numTxAntennas);
        
        % 信号通过信道
        [rxSignal, csi] = apply_channel(tdlChannel, txSignal);
        
        % 存储数据
        txSignals{i} = txSignal;
        rxSignals{i} = rxSignal;
        channelMatrices{i} = csi;
    end

    % 保存数据集
    save_dataset(config.savePath, txSignals, rxSignals, channelMatrices);
    fprintf('Dataset saved to %s\n', config.savePath);
end
