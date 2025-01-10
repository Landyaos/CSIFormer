%% 参数设置
% 系统参数配置
SNR = 20;                                                 % 信噪比
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
    'PathGainsOutputPort', true);   % 开启路径增益输出

% 评价体系
errorRate = comm.ErrorRate;

% 超参数
batchSize = 50000;
% 数据定义
txSignalData = zeros(batchSize, numValidSubc, numSym, numTx, 2);
rxSignalData = zeros(batchSize, numValidSubc, numSym, numRx, 2);
csiData = zeros(batchSize, numValidSubc, numSym, numTx, numRx, 2);
txPilotSignalData = zeros(batchSize, numValidSubc, numSym, numTx, 2);
rxPilotSignalData = zeros(batchSize, numValidSubc, numSym, numRx, 2);

for i = 1:batchSize
    %% 数据发送与接收
    % 数据符号生成
    txSymStream = randi([0 M-1], numSubFrameSym, 1); 
    % 调制成符号
    dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式

    % 重塑数据符号为所需维度
    dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
    
    % 导频符号生成
    pilotQPSKSymbols = [1+1i, 1+1i, 1+1i, 1+1i];
    pilotSignal = pilotQPSKSymbols(randi(length(pilotQPSKSymbols), numPilot, numSym, numTx));
    
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
    
    % CSI矩阵
    mimoChannelInfo = info(mimoChannel);
    pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
    toffset = mimoChannelInfo.ChannelFilterDelay;
    h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset); % Nsc x Nsym x Nt x Nr
    
    % 噪声
    [transmitSignal, noiseVar] = awgn(transmitSignal, SNR, "measured");
    
    % OFDM 解调
    [rxDataSignal, rxPilotSignal] = ofdmDemod(transmitSignal);
    
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
    csiData(i,:,:,:,:,1) = real(h);
    csiData(i,:,:,:,:,2) = imag(h);
    % 发送信号（实部和虚部分离）
    txSignalData(i,:,:,:,1) = real(originSignal(validSubcIndices, :, :));
    txSignalData(i,:,:,:,2) = imag(originSignal(validSubcIndices, :, :));
    % 接收信号（实部和虚部分离）
    rxSignalData(i,:,:,:,1) = real(rxSignal(validSubcIndices, :, :));
    rxSignalData(i,:,:,:,2) = imag(rxSignal(validSubcIndices, :, :));

    % 发送导频信号（实部和虚部分离）
    txPilotOverallSignal = zeros(numSubc, numSym, numTx);
    for tx = 1:numTx
        for sym = 1:numSym
            txPilotOverallSignal(pilotIndices(:,sym,tx), sym, tx) = pilotSignal(:, sym, tx);
        end
    end
    txPilotSignalData(i,:,:,:,1) = real(txPilotOverallSignal(validSubcIndices,:,:)); % 实部
    txPilotSignalData(i,:,:,:,2) = imag(txPilotOverallSignal(validSubcIndices,:,:)); % 虚部
    
    % 接收导频符号（实部和虚部分离）
    rxPilotOverallSignal = zeros(numSubc, numSym, numRx);
    for rx = 1:numRx
        for tx = 1:numTx
            for sym = 1:numSym
            rxPilotOverallSignal(pilotIndices(:,sym,tx),sym,rx) = rxPilotSignal(:,sym,tx,rx);
            end
        end
    end
    rxPilotSignalData(i,:,:,:,1) = real(rxPilotOverallSignal(validSubcIndices,:,:)); % 实部
    rxPilotSignalData(i,:,:,:,2) = imag(rxPilotOverallSignal(validSubcIndices,:,:)); % 虚部
end

dataIndicesData = dataIndices-numGuardBands(1)-1; % 从0开始
pilotIndicesData = pilotIndices-numGuardBands(1)-1; % 从0开始
% 保存批量数据到文件
save('trainDataTrain.mat', ...
    'csiData', ...
    'txSignalData',...
    'rxSignalData',...
    'rxPilotSignalData', ...
    'txPilotSignalData',...
    'dataIndicesData',...
    'pilotIndicesData',...
    '-v7.3');

