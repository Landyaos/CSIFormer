clear;
clc;
%% 参数设置
% 系统参数配置
snrValues = 0:5:30;                                      % 信噪比范围
numSubc = 64;                                             % FFT 长度
numGuardBands = [6;6];                                    % 左右保护带
numPilot = 4;                                             % 每根天线的导频子载波
numTx = 2;                                                % 发射天线数量
numRx = 2;                                                % 接收天线数量
numSym = 14;                                              % 每帧 OFDM 符号数
numStream = 2;                                            % 数据流个数
cpLength = 16;                                            % 循环前缀长度

% 调制参数配置
M = 2;                                                    % QPSK 调制（M=2）

% 信道模型配置
sampleRate = 15.36e6;                                     % 采样率
pathDelays = [0 0.5e-6];                                  % 路径时延
averagePathGains = [0 -2];                                % 平均路径增益
maxDopplerShift = 200;                                    % 最大多普勒频移

% 信道估计配置
CEC.pilotAverage = 'UserDefined';
CEC.freqWindow = 3;
CEC.timeWindow = 3;
CEC.interpType = 'linear';
CEC.algorithm = 'ls';

% 信号导频分布配置
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


% OFDM解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubc, ...
                                  'NumGuardBandCarriers', numGuardBands, ...
                                  'NumSymbols', numSym, ...
                                  'PilotOutputPort', true, ...
                                  'PilotCarrierIndices', pilotIndices, ...
                                  'CyclicPrefixLength', cpLength, ...
                                  'NumReceiveAntennas', numRx);
% OFDM调制器
% ofdmMod = comm.OFDMModulator('FFTLength', numSubc, ...
%                              'NumGuardBandCarriers', numGuardBands, ...
%                              'NumSymbols', numSym, ...
%                              'PilotInputPort', true, ...
%                              'PilotCarrierIndices', pilotIndices, ...
%                              'CyclicPrefixLength', cpLength, ...
%                              'NumTransmitAntennas', numTx);
ofdmMod = comm.OFDMModulator(ofdmDemod);

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
    'Seed', 10086, ... % 固定随机种子
    'PathGainsOutputPort', true);   % 开启路径增益输出

% 简单信道
% mimoChannel = comm.MIMOChannel(...
%     'SampleRate', sampleRate, ...
%     'SpatialCorrelationSpecification', 'None',...
%     'NumTransmitAntennas', numTx, ...
%     'NumReceiveAntennas', numRx, ...
%     'FadingDistribution', 'Rayleigh', ...
%     'RandomStream', 'mt19937ar with seed', ...
%     'Seed', 123, ... % 固定随机种子
%     'PathGainsOutputPort', true);   % 开启路径增益输出

% 评价体系
errorRate = comm.ErrorRate;

%% 数据集采集
numFrame = 2;
datasetPath = {'../raw/eqTrainData.mat','../raw/eqValData.mat'};
datasetConfig = [2000,500];


for datasetIdx = 1:length(datasetPath)
    snrDatasetSize = datasetConfig(datasetIdx);
    
    datasetCapacity = snrDatasetSize * length(snrValues);
    txSignalData = zeros(datasetCapacity, numValidSubc, numSym, numTx, 2);
    rxSignalData = zeros(datasetCapacity, numValidSubc, numSym, numRx, 2);
    csiLSData = zeros(datasetCapacity, numValidSubc, numSym, numTx, numRx, 2);
    csiLabelData = zeros(datasetCapacity, numValidSubc, numSym, numTx, numRx, 2);
    csiPreData = zeros(datasetCapacity, numFrame, numValidSubc, numSym, numTx, numRx, 2);
    
    csiPreTemp = zeros(numFrame+1, numValidSubc, numSym, numTx, numRx, 2);
    for snrIdx = 1:length(snrValues)
        snrIdx
        snr = snrValues(snrIdx);
        for frame = -1:snrDatasetSize
            % 数据符号生成
            txSymStream = randi([0 M-1], numSubFrameSym, 1); 
            dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式
            dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
            pilotSignal = pilotSymbols(randi(length(pilotSymbols), numPilot, numSym, numTx));
            
            originSignal = zeros(numSubc, numSym, numTx);
            originSignal(dataIndices, :, :) = dataSignal;
            for tx = 1:numTx
                originSignal(pilotIndices(:,1,tx),:,tx) = pilotSignal(:, :, tx);
            end    
            

            % OFDM 调制
            txSignal = ofdmMod(dataSignal, pilotSignal); % 结果为 (80 × 14 × 2)，包含循环前缀的时域信号
            % 通过信道模型获取接收信号和路径增益[总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
            [transmitSignal, pathGains] = mimoChannel(txSignal);
            % 噪声
            [transmitSignal, noiseVar] = awgn(transmitSignal, snr, "measured");
            % OFDM 解调
            [rxDataSignal, rxPilotSignal] = ofdmDemod(transmitSignal);
            

            % 完美CSI矩阵Nsc x Nsym x Nt x Nr
            mimoChannelInfo = info(mimoChannel);
            pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
            toffset = mimoChannelInfo.ChannelFilterDelay;
            h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset);
            
            csiPreTemp(1,:,:,:,:,:) = csiPreTemp(2,:,:,:,:,:);
            csiPreTemp(2,:,:,:,:,:) = csiPreTemp(3,:,:,:,:,:);
            csiPreTemp(3,:,:,:,:,1) = real(h);
            csiPreTemp(3,:,:,:,:,2) = imag(h);
            
            %% 信道均衡
            % ToolBox 函数均衡
            % hPerfect = h(dataIndices-6,:,:,:);
            % hReshaped = reshape(hPerfect,[],numTx,numRx);
            % eqSignal = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="mmse");
            % eqSignal = reshape(eqSignal, [], 1);  
            % eqStream = pskdemod(eqSignal, M);
            % 
            % errorRate = comm.ErrorRate;
            % BER_perfect = errorRate(txSymStream, eqStream);
            % fprintf('\n Perfect Symbol error rate = %d from %d errors in %d symbols\n',BER_perfect);

            if frame < 1
                continue;
            end
            % 接收信号
            finalSignal = zeros(numSubc, numSym, numRx);
            finalSignal(dataIndices, :, :) = rxDataSignal(:,:,:);
            for rx = 1:numRx
                for tx = 1:numTx
                    finalSignal(pilotIndices(:,1,tx),:,rx) = rxPilotSignal(:,:,tx,rx);
                end
            end

            % 计算导频处信道估计
            csiLS = zeros(numSubc, numSym, numTx, numRx);
            for tx = 1:numTx
                for rx = 1:numRx
                    csiLS(pilotIndices(:,1,tx),:,tx,rx) = rxPilotSignal(:,:,tx,rx) ./ pilotSignal(:, :, tx);
                end
            end
            
        
            %% 数据保存
            dataIdx = snrDatasetSize * (snrIdx-1) + frame;
            dataIdx;
            % csi_pre
            csiPreData(dataIdx,:,:,:,:,:,:) = csiPreTemp(1:2,:,:,:,:,:);
            % csi
            csiLabelData(dataIdx,:,:,:,:,:) = csiPreTemp(3,:,:,:,:,:);
            % csi_ls
            csiLSData(dataIdx,:,:,:,:,1) = real(csiLS(validSubcIndices,:,:,:));
            csiLSData(dataIdx,:,:,:,:,2) = imag(csiLS(validSubcIndices,:,:,:));
            % 发送信号（实部和虚部分离）
            txSignalData(dataIdx,:,:,:,1) = real(originSignal(validSubcIndices, :, :));
            txSignalData(dataIdx,:,:,:,2) = imag(originSignal(validSubcIndices, :, :));
            % 接收信号（实部和虚部分离）
            rxSignalData(dataIdx,:,:,:,1) = real(finalSignal(validSubcIndices, :, :));
            rxSignalData(dataIdx,:,:,:,2) = imag(finalSignal(validSubcIndices, :, :));
            
        end
    end    
    disp('save data ...')
    test = reshape(1:(2*3*4*5), [2, 3, 4, 5]);

    % 保存批量数据到文件
    save(datasetPath{datasetIdx}, ...
        'csiLabelData', ...
        'txSignalData',...
        'rxSignalData',...
        'test',...
        '-v7.3');
end


