%% 参数设置
% 系统参数配置
snr = 15;                                                 % 信噪比
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
maxDopplerShift = 300;                                     % 最大多普勒频移

% 信道估计配置
CEC.pilotAverage = 'UserDefined';
CEC.freqWindow = 3;
CEC.timeWindow = 3;
CEC.interpType = 'linear';
CEC.algorithm = 'ls';


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

%% 数据发送与接收

% 数据符号生成
txSymStream = randi([0 M-1], numSubFrameSym, 1); 
% 调制成符号
dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式
disp(size(dataSignal))
% 重塑数据符号为所需维度
dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);

% 导频符号生成
pilotQPSKSymbols = [1+1i, 1+1i, 1+1i, 1+1i];
pilotSignal = pilotQPSKSymbols(randi(length(pilotQPSKSymbols), numPilot, numSym, numTx));

% OFDM 调制
txSignal = ofdmMod(dataSignal, pilotSignal); % 结果为 (80 × 14 × 2)，包含循环前缀的时域信号


% 通过信道模型获取接收信号和路径增益
[transmitSignal, pathGains] = mimoChannel(txSignal); % pathGains: [总样本数, N_path, numTransmitAntennas, numReceiveAntennas]

% CSI矩阵
mimoChannelInfo = info(mimoChannel);
pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
toffset = mimoChannelInfo.ChannelFilterDelay;
h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, dataIndices, toffset); % Nsc x Nsym x Nt x Nr

% 噪声
[rxSignal, noiseVar] = awgn(transmitSignal, snr, "measured");

% OFDM 解调
[rxDataSignal, rxPilotSignal] = ofdmDemod(rxSignal);


csi = zeros(numDataSubc, numSym, numTx, numRx);

for sym = 1:numSym
    for subc = 1:numDataSubc
        csi(subc,sym, :, :) = rxDataSignal(subc, sym, :) ./ dataSignal(subc, sym, :);
    end
end


% 信道估计
hEst = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);
% 信道均衡
hReshaped = reshape(csi,[],numTx,numRx);
equalizedSymbols = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="zf");

% eq = zeros(numDataSubc, numSym, numRx)
% for tx = 1:numTx
%     for rx = 1:numRx
% 
%     end
% end


% 将接收到的数据符号转换为列向量
equalizedSymbols = reshape(equalizedSymbols, [], 1);  
% 解调接收数据符号
rxSymStream = pskdemod(equalizedSymbols, M);

%% 评价
% 误码率计算
BER = errorRate(txSymStream, rxSymStream);
fprintf('\nSymbol error rate = %d from %d errors in %d symbols\n',BER);

%% 自定义函数

function [H_est] = channelEstimate(rxPilotSignal,refPilotSignal, dataIndices, pilotIndices, CEC)
    % MIMO-OFDM 信道估计函数
    % 输入：
    %   rxDataSignal: 接收信号矩阵 (numDataSubc x numSym x numRx)
    %   rxPilotSignal: 接收导频信号(numPilotSubc x numSym x numTx x numRx)
    %   refPilotSignal: 参考导频信号 (numPilotSubc x numSym x numTx)
    %   dataIndices: 数据符号索引
    %   pilotIndices: 导频符号索引(numPilotSubc x numSym x numTx)
    %   CEC.algorithm: 估计算法 ('LS' 或 'MMSE')
    %   CEC.interpType: 插值类型 ('nearest', 'linear', 'cubic', 'spline')
    %   CEC.freqWindow: 频域平均窗口大小;
    %   CEC.timeWindow: 时域平均窗口大小;T
    %   
    % 输出：
    %   H_est: 估计的信道响应矩阵 (numSubc x numSym x numTx x numRx)
    %   noiseVar: 噪声平均功率

    % 提取信号维度
    numDataSubc = length(dataIndices);
    [~, numSym, numTx, numRx] = size(rxPilotSignal);

    % 初始化估计的信道响应矩阵和噪声功率
    H_est = zeros(numDataSubc, numSym, numTx, numRx);

    for tx = 1:numTx
        for rx = 1:numRx
            % 获取当前发射 - 接收天线对的导频信号
            pilotRxSignal = rxPilotSignal(:,:,tx,rx);
            pilotRefSignal = refPilotSignal(:, :, tx);
            H_ls = pilotRxSignal./ pilotRefSignal; % []
            % 导频平均
            H_avg = pilotAveraging(H_ls, CEC.freqWindow, CEC.timeWindow);
            % 信道插值
            [X, Y] = meshgrid(1:numSym, dataIndices);
            [Xp, Yp] = meshgrid(1:numSym,squeeze(pilotIndices(:,1,tx)));
            H_est(:, :, tx, rx)  = griddata(Xp, Yp, H_avg, X, Y);
        end
    end
end

function H_avg = pilotAveraging(H_ls, freqWindow, timeWindow)
    [numPilotSubc, numSym] = size(H_ls);
    H_avg = zeros(size(H_ls));

    halfFreqWindow = floor(freqWindow / 2);
    halfTimeWindow = floor(timeWindow / 2);

    for sym = 1:numSym
        for subc = 1:numPilotSubc
            startSubc = max(1, subc - halfFreqWindow);
            endSubc = min(numPilotSubc, subc + halfFreqWindow);
            startSym = max(1, sym - halfTimeWindow);
            endSym = min(numSym, sym + halfTimeWindow);

            window = H_ls(startSubc:endSubc, startSym:endSym);
            H_avg(subc, sym) = mean(window(:));
        end
    end
end
