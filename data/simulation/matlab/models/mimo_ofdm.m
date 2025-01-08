% 参数设置
numSubcarriers = 64;                                      % FFT 长度
numGuardBands = [6;6];                                    % 左右保护带
numPilots = 4;                                            % 每根天线的导频子载波
numTransmitAntennas = 2;                                  % 发射天线数量
numReceiveAntennas = 2;                                   % 接收天线数量
numSymbols = 14;                                          % 每帧 OFDM 符号数
cpLength = 16;                                            % 循环前缀长度
modulationOrder = 4;                                      % QPSK 调制（M=4）
bitsPerSymbol = log2(modulationOrder);
snr = 20 ;

sampleRate = 15.36e6;                                     % 采样率
% pathDelays = [0 0.5e-6 1.2e-6];                           % 路径时延
% averagePathGains = [0 -2 -5];                             % 平均路径增益
maxDopplerShift = 50;                                     % 最大多普勒频移

% 初始化有效子载波范围
validSubcarrierRange = (numGuardBands(1)+1):(numSubcarriers-numGuardBands(2));
numValidSubcarriers = length(validSubcarrierRange);


% 导频索引 (满足范围限制)
pilotIndicesAntenna1 = [12; 26; 40; 54]; % 天线 1 导频索引
pilotIndicesAntenna2 = [13; 27; 41; 55]; % 天线 2 导频索引
% 构造 PilotCarrierIndices (3D 矩阵, NPilot-by-NSym-by-NT)
pilotIndices = zeros(numPilots, numSymbols, numTransmitAntennas);
pilotIndices(:, :, 1) = repmat(pilotIndicesAntenna1, 1, numSymbols); % 天线 1
pilotIndices(:, :, 2) = repmat(pilotIndicesAntenna2, 1, numSymbols); % 天线 2
% 随机 QPSK 导频符号
pilotQPSKSymbols = [1+1i, 1+1i, 1+1i, 1+1i];
pilotSymbols = pilotQPSKSymbols(randi(length(pilotQPSKSymbols), numPilots, numSymbols, numTransmitAntennas));

% 数据子载波数量
numDataSubcarriers = numSubcarriers-sum(numGuardBands)-(numPilots*numTransmitAntennas);
dataSymbolIndices = setdiff(validSubcarrierRange,unique([pilotIndicesAntenna1; pilotIndicesAntenna2]));


%% 发射机

% 比特流生成
numBits = numDataSubcarriers * numSymbols * bitsPerSymbol;
bitStream = randi([0 1], numBits, 1);  % 随机生成比特流
% 调制成符号
dataSymbols = pskmod(bitStream, modulationOrder, pi/4, 'gray');  % 调制后的符号为复数形式
% 重塑数据符号为所需维度
dataSymbols = reshape(dataSymbols, numDataSubcarriers, numSymbols, numTransmitAntennas);

% OFDM调制器
ofdmMod = comm.OFDMModulator('FFTLength', numSubcarriers, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'NumSymbols', numSymbols, ...
                             'PilotInputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumTransmitAntennas', numTransmitAntennas);
% OFDM 调制
txSignal = ofdmMod(dataSymbols, pilotSymbols); % 结果为 (80 × 14 × 2)，包含循环前缀的时域信号

% MIMO信道模型
channelModel = comm.MIMOChannel(...
    'SampleRate', sampleRate, ...
    'SpatialCorrelationSpecification', 'None',...
    'NumTransmitAntennas', numTransmitAntennas, ...
    'NumReceiveAntennas', numReceiveAntennas, ...
    'FadingDistribution', 'Rayleigh', ...
    'PathGainsOutputPort', true);   % 开启路径增益输出

% 噪声模型
awgnChannel = comm.AWGNChannel( ...
    'NoiseMethod', 'Signal to noise ratio (SNR)', ...
    'SNR', snr); % 设置 SNR 为 20 dB
rxSignal = awgnChannel(rxSignal);

%% 接收机

% 创建 OFDM 解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubcarriers, ...
                                  'NumGuardBandCarriers', numGuardBands, ...
                                  'NumSymbols', numSymbols, ...
                                  'PilotOutputPort', true, ...
                                  'PilotCarrierIndices', pilotIndices, ...
                                  'CyclicPrefixLength', cpLength, ...
                                  'NumReceiveAntennas', numReceiveAntennas);

% OFDM 解调
[rxDataSymbols, rxPilotSymbols] = ofdmDemod(rxSignal);

%% 信道估计
% 初始化估计的 CSI 矩阵
estimatedCSI = zeros(numReceiveAntennas, numTransmitAntennas, numDataSubcarriers, numSymbols);
estimatedPilotCSI = zeros(numReceiveAntennas, numTransmitAntennas, numPilots, numSymbols);
% 基于导频符号的信道估计
for symIdx = 1:numSymbols
    for rx = 1:numReceiveAntennas
        for tx = 1:numTransmitAntennas
            % 提取接收导频符号
            receivedPilots = rxPilotSymbols(:, symIdx, tx, rx); % [numPilots, 1]
            transmittedPilots = pilotSymbols(:, symIdx, tx);    % [numPilots, 1]
            
            % 估计导频子载波上的信道响应
            pilotCSI = receivedPilots ./ transmittedPilots; % [numPilots, 1]
            estimatedPilotCSI(rx, tx, :, symIdx) = pilotCSI;

            % 插值到所有有效子载波
            pilotIndicesRange = pilotIndices(:, symIdx, tx); % [numPilots, 1]
            estimatedCSI(rx, tx, :, symIdx) = interp1(pilotIndicesRange, ...
                                                      pilotCSI, ...
                                                      dataSymbolIndices, ...
                                                      'linear', 'extrap');
        end
    end
end

% 均衡
% 初始化均衡后的数据符号矩阵
equalizedSymbols = zeros(numDataSubcarriers, numSymbols, numTransmitAntennas);

% 对每个符号进行均衡
for symIdx = 1:numSymbols
    for subIdx = 1:numDataSubcarriers
        % 构造当前子载波的信道矩阵 H_sub (接收天线 x 发射天线)
        H_sub = squeeze(estimatedCSI(:, :, subIdx, symIdx)); % [numReceiveAntennas, numTransmitAntennas]

        % 提取接收的符号向量 y_sub (接收天线 x 1)
        y_sub = squeeze(rxDataSymbols(subIdx, symIdx, :)); % [numReceiveAntennas, 1]
        disp(size(y_sub))
        % 零强迫均衡: W = (H^H * H)^-1 * H^H
        W = pinv(H_sub); % [numTransmitAntennas, numReceiveAntennas]

        % 均衡后的符号: s = W * y
        equalizedSymbols(subIdx, symIdx, :) = W * y_sub; % [numTransmitAntennas, 1]
    end
end

% 数据符号解调
% 将接收到的数据符号转换为列向量
equalizedSymbols = reshape(equalizedSymbols, [], 1);  

% 原始比特流
originalBits = bitStream;

% 解调接收数据符号
receivedBits = pskdemod(equalizedSymbols, modulationOrder, pi/4, 'gray');

% 误码率计算
% 确保发送和接收比特流维度一致
originalBits = originalBits(1:length(receivedBits));
numErrors = sum(originalBits ~= receivedBits);  % 计算比特错误数量
ber = numErrors / length(originalBits);         % 计算误码率

% 输出 BER
fprintf('误码率 (BER): %.6f\n', ber);

