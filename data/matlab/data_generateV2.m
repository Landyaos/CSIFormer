%% 参数设置
% 系统参数配置
numSubFrame = 10;                                         % 子帧数量
snrValues = 0:5:30;                                       % 信噪比范围
numSubc = 64;                                             % FFT 长度
numGuardBands = [6;6];                                    % 左右保护带
numPilot = 4;                                             % 每根天线的导频子载波
numTx = 2;                                                % 发射天线数量
numRx = 2;                                                % 接收天线数量
numSym = 14;                                              % 每帧 OFDM 符号数
numStream = 2;                                            % 数据流个数
cpLength = 16;                                            % 循环前缀长度

% 调制参数配置
M = 2;                                                    % QPSK 调制（M=4）

% 信道模型配置
sampleRate = 15.36e6;                                     % 采样率
pathDelays = [0 0.5e-6 1.2e-6];                           % 路径时延
averagePathGains = [0 -2 -5];                             % 平均路径增益
maxDopplerShift = 50;                                     % 最大多普勒频移

% 信道估计配置
CEC.pilotAverage = 'UserDefined';
CEC.freqWindow = 3;
CEC.timeWindow = 3;
CEC.interpType = 'linear';
CEC.algorithm = 'ls';

validSubcIndices = (numGuardBands(1)+1):(numSubc-numGuardBands(2));
numValidSubc = length(validSubcIndices);


% 导频子载波配置
pilotIndicesAnt1 = [7; 26; 40; 57]; % 天线 1 导频索引
pilotIndicesAnt2 = [8; 27; 41; 58]; % 天线 2 导频索引
% 构造 PilotCarrierIndices (3D 矩阵, NPilot-by-NSym-by-NT)
pilotIndices = zeros(numPilot, numSym, numTx);
pilotIndices(:, :, 1) = repmat(pilotIndicesAnt1, 1, numSym); % 天线 1
pilotIndices(:, :, 2) = repmat(pilotIndicesAnt2, 1, numSym); % 天线 2
% 导频集合
pilotSymbols = [1+1i, 1+1i, 1+1i, 1+1i];
% 数据子载波配置
dataIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)),unique(pilotIndices));
numDataSubc = length(dataIndices);
numSubFrameSym = numDataSubc * numSym * numTx;

% OFDM调制器
ofdmMod = comm.OFDMModulator('FFTLength', numSubc, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'NumSymbols', numSym, ...
                             'PilotInputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumTransmitAntennas', numTx);
% OFDM解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubc, ...
                                  'NumGuardBandCarriers', numGuardBands, ...
                                  'NumSymbols', numSym, ...
                                  'PilotOutputPort', true, ...
                                  'PilotCarrierIndices', pilotIndices, ...
                                  'CyclicPrefixLength', cpLength, ...
                                  'NumReceiveAntennas', numRx);
% 信道模型
mimoChannel = comm.MIMOChannel(...
    'SampleRate', sampleRate, ...
    'SpatialCorrelationSpecification', 'None',...
    'PathDelays', pathDelays, ...
    'AveragePathGains', averagePathGains, ...
    'MaximumDopplerShift', maxDopplerShift, ...
    'NumTransmitAntennas', numTx, ...
    'NumReceiveAntennas', numRx, ...
    'FadingDistribution', 'Rayleigh', ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 123, ... % 固定随机种子
    'PathGainsOutputPort', true);   % 开启路径增益输出
% 评价体系
errorRate = comm.ErrorRate;

% 超参数
% 训练集
snrTrainSize = 50000;
dataTrainSize = snrTrainSize * length(snrValues);
txSignalTrainData = zeros(dataTrainSize, numValidSubc, numSym, numTx, 2);
rxSignalTrainData = zeros(dataTrainSize, numValidSubc, numSym, numRx, 2);
csiTrainData = zeros(dataTrainSize, numValidSubc, numSym, numTx, numRx, 2);
% 验证集
snrValSize = 5000;
dataValSize = snrValSize * length(snrValues);
txSignalValData = zeros(dataValSize, numValidSubc, numSym, numTx, 2);
rxSignalValData = zeros(dataValSize, numValidSubc, numSym, numRx, 2);
csiValData = zeros(dataValSize, numValidSubc, numSym, numTx, numRx, 2);
% 测试集
snrTestSize = 5000;
dataTestSize = snrTestSize * length(snrValues);
txSignalTestData = zeros(dataTestSize, numValidSubc, numSym, numTx, 2);
rxSignalTestData = zeros(dataTestSize, numValidSubc, numSym, numRx, 2);
csiTestData = zeros(dataTestSize, numValidSubc, numSym, numTx, numRx, 2);

for snrIdx = 1:length(snrValues)
    snr = snrValues(snrIdx);
    for step = 1:snrTrainSize
        dataIdx = snrTrainSize * (snrIdx-1) + step;
        
        %% 数据发送与接收
        % 数据符号生成
        txSymStream = randi([0 M-1], numSubFrameSym, 1); 
        % 调制成符号
        dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式
        % 重塑数据符号为所需维度
        dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
        % 导频符号
        pilotSignal = pilotSymbols(randi(length(pilotSymbols), numPilot, numSym, numTx));
        originSignal = zeros(numSubc, numSym, numTx);
        originSignal(dataIndices, :, :) = dataSignal;
        for tx = 1:numTx
            for sym = 1:numSym
                originSignal(pilotIndices(:,sym,tx),sym,tx) = pilotSignal(:, sym, tx);
            end
        end    
        % OFDM 调制
        txSignal = ofdmMod(dataSignal, pilotSignal); % 结果为 (80 × 14 × 2)，包含循环前缀的时域信号
        % 通过信道模型获取接收信号和路径增益
        [transmitSignal, pathGains] = mimoChannel(txSignal); % pathGains: [总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
        % 噪声
        [transmitSignal, noiseVar] = awgn(transmitSignal, snr, "measured");
        % OFDM 解调
        [rxDataSignal, rxPilotSignal] = ofdmDemod(transmitSignal);

        % 完美CSI矩阵
        mimoChannelInfo = info(mimoChannel);
        pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
        toffset = mimoChannelInfo.ChannelFilterDelay;
        h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset); % Nsc x Nsym x Nt x Nr

        rxSignal = zeros(numSubc, numSym, numRx);
        rxSignal(dataIndices, :, :) = rxDataSignal(:,:,:);
        for rx = 1:numRx
            for tx = 1:numTx
                for sym = 1:numSym
                rxSignal(pilotIndices(:,sym,tx),sym,rx) = rxPilotSignal(:,sym,tx,rx);
                end
            end
        end
        
        %% 数据保存
        % csi
        csiTrainData(dataIdx,:,:,:,:,1) = real(h);
        csiTrainData(dataIdx,:,:,:,:,2) = imag(h);
        % 发送信号（实部和虚部分离）
        txSignalTrainData(dataIdx,:,:,:,1) = real(originSignal(validSubcIndices, :, :));
        txSignalTrainData(dataIdx,:,:,:,2) = imag(originSignal(validSubcIndices, :, :));
        % 接收信号（实部和虚部分离）
        rxSignalTrainData(dataIdx,:,:,:,1) = real(rxSignal(validSubcIndices, :, :));
        rxSignalTrainData(dataIdx,:,:,:,2) = imag(rxSignal(validSubcIndices, :, :));

    end

    for step = 1:snrValSize

        dataIdx = snrValSize * (snrIdx-1) + step;
        
        %% 数据发送与接收
        % 数据符号生成
        txSymStream = randi([0 M-1], numSubFrameSym, 1); 
        % 调制成符号
        dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式
        % 重塑数据符号为所需维度
        dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
        % 导频符号
        pilotSignal = pilotSymbols(randi(length(pilotSymbols), numPilot, numSym, numTx));
        originSignal = zeros(numSubc, numSym, numTx);
        originSignal(dataIndices, :, :) = dataSignal;
        for tx = 1:numTx
            for sym = 1:numSym
                originSignal(pilotIndices(:,sym,tx),sym,tx) = pilotSignal(:, sym, tx);
            end
        end    
        % OFDM 调制
        txSignal = ofdmMod(dataSignal, pilotSignal); % 结果为 (80 × 14 × 2)，包含循环前缀的时域信号
        % 通过信道模型获取接收信号和路径增益
        [transmitSignal, pathGains] = mimoChannel(txSignal); % pathGains: [总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
        % 噪声
        [transmitSignal, noiseVar] = awgn(transmitSignal, snr, "measured");
        % OFDM 解调
        [rxDataSignal, rxPilotSignal] = ofdmDemod(transmitSignal);

        % 完美CSI矩阵
        mimoChannelInfo = info(mimoChannel);
        pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
        toffset = mimoChannelInfo.ChannelFilterDelay;
        h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset); % Nsc x Nsym x Nt x Nr

        rxSignal = zeros(numSubc, numSym, numRx);
        rxSignal(dataIndices, :, :) = rxDataSignal(:,:,:);
        for rx = 1:numRx
            for tx = 1:numTx
                for sym = 1:numSym
                rxSignal(pilotIndices(:,sym,tx),sym,rx) = rxPilotSignal(:,sym,tx,rx);
                end
            end
        end
        
        %% 数据保存
        % csi
        csiValData(dataIdx,:,:,:,:,1) = real(h);
        csiValData(dataIdx,:,:,:,:,2) = imag(h);
        % 发送信号（实部和虚部分离）
        txSignalValData(dataIdx,:,:,:,1) = real(originSignal(validSubcIndices, :, :));
        txSignalValData(dataIdx,:,:,:,2) = imag(originSignal(validSubcIndices, :, :));
        % 接收信号（实部和虚部分离）
        rxSignalValData(dataIdx,:,:,:,1) = real(rxSignal(validSubcIndices, :, :));
        rxSignalValData(dataIdx,:,:,:,2) = imag(rxSignal(validSubcIndices, :, :));        
    
    end

    for step = 1:snrTestSize
        dataIdx = snrTestSize * (snrIdx-1) + step;
        
        %% 数据发送与接收
        % 数据符号生成
        txSymStream = randi([0 M-1], numSubFrameSym, 1); 
        % 调制成符号
        dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式
        % 重塑数据符号为所需维度
        dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
        % 导频符号
        pilotSignal = pilotSymbols(randi(length(pilotSymbols), numPilot, numSym, numTx));
        originSignal = zeros(numSubc, numSym, numTx);
        originSignal(dataIndices, :, :) = dataSignal;
        for tx = 1:numTx
            for sym = 1:numSym
                originSignal(pilotIndices(:,sym,tx),sym,tx) = pilotSignal(:, sym, tx);
            end
        end    
        % OFDM 调制
        txSignal = ofdmMod(dataSignal, pilotSignal); % 结果为 (80 × 14 × 2)，包含循环前缀的时域信号
        % 通过信道模型获取接收信号和路径增益
        [transmitSignal, pathGains] = mimoChannel(txSignal); % pathGains: [总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
        % 噪声
        [transmitSignal, noiseVar] = awgn(transmitSignal, snr, "measured");
        % OFDM 解调
        [rxDataSignal, rxPilotSignal] = ofdmDemod(transmitSignal);

        % 完美CSI矩阵
        mimoChannelInfo = info(mimoChannel);
        pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
        toffset = mimoChannelInfo.ChannelFilterDelay;
        h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset); % Nsc x Nsym x Nt x Nr

        rxSignal = zeros(numSubc, numSym, numRx);
        rxSignal(dataIndices, :, :) = rxDataSignal(:,:,:);
        for rx = 1:numRx
            for tx = 1:numTx
                for sym = 1:numSym
                rxSignal(pilotIndices(:,sym,tx),sym,rx) = rxPilotSignal(:,sym,tx,rx);
                end
            end
        end
        
        %% 数据保存
        % csi
        csiTestData(dataIdx,:,:,:,:,1) = real(h);
        csiTestData(dataIdx,:,:,:,:,2) = imag(h);
        % 发送信号（实部和虚部分离）
        txSignalTestData(dataIdx,:,:,:,1) = real(originSignal(validSubcIndices, :, :));
        txSignalTestData(dataIdx,:,:,:,2) = imag(originSignal(validSubcIndices, :, :));
        % 接收信号（实部和虚部分离）
        rxSignalTestData(dataIdx,:,:,:,1) = real(rxSignal(validSubcIndices, :, :));
        rxSignalTestData(dataIdx,:,:,:,2) = imag(rxSignal(validSubcIndices, :, :));             
    
    end


end


% 保存批量数据到文件
save('../raw/trainData.mat', ...
    'csiTrainData', ...
    'txSignalTrainData',...
    'rxSignalTrainData',...
    '-v7.3');

% 保存验证集批量数据到文件
save('../raw/valData.mat', ...
    'csiValData', ...
    'txSignalValData', ...
    'rxSignalValData', ...
    '-v7.3');

% 保存测试集批量数据到文件
save('../raw/testData.mat', ...
    'csiTestData', ...
    'txSignalTestData', ...
    'rxSignalTestData', ...
    '-v7.3');

