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
snr = 30;

sampleRate = 15.36e6;                                     % 采样率
pathDelays = [0 0.5e-6 1.2e-6];                           % 路径时延
averagePathGains = [0 -2 -5];                             % 平均路径增益
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


%% OFDM调制

% 比特流生成
numBits = numDataSubcarriers * numSymbols * bitsPerSymbol;
originalBits = randi([0 1], numBits, 1);  % 随机生成比特流
% 调制成符号
dataSymbols = pskmod(originalBits, modulationOrder, pi/4);  % 调制后的符号为复数形式
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

%% MIMO信道模型
% 信号模型初始化
channelModel = comm.MIMOChannel(...
    'SampleRate', sampleRate, ...
    'SpatialCorrelationSpecification', 'None',...
    'NumTransmitAntennas', numTransmitAntennas, ...
    'NumReceiveAntennas', numReceiveAntennas, ...
    'FadingDistribution', 'Rayleigh', ...
    'PathGainsOutputPort', true);   % 开启路径增益输出

% 通过信道模型获取接收信号和路径增益
[rxSignal, pathGains] = channelModel(txSignal); % pathGains: [总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
% 噪声
awgnChannel = comm.AWGNChannel(...
    'NoiseMethod', 'Signal to noise ratio (SNR)', ...
    'SNR', snr); % 设置 SNR 为 20 dB

% rxSignal = awgnChannel(rxSignal);


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


pathGains = squeeze(pathGains(:,1,:,:));
pathGains = reshape(pathGains, numSubcarriers+cpLength, numSymbols, numTransmitAntennas, numReceiveAntennas);
pathGains = pathGains(cpLength+1:end,:,:,:);
pathGains = fft(pathGains, numSubcarriers, 1);
pathGains = fftshift(pathGains, 1);
pathGains = pathGains(dataSymbolIndices, :, :, :);

disp(size(txSignal))

for i = 1:1:44
    for j = 1:1:1
        disp(reshape(rxDataSymbols(i,j,:),1,2));
        disp(reshape(dataSymbols(i,j,:),1,2)*squeeze(pathGains(i,j,:,:)));
    end
end


%% 评价体系

errorRate = comm.ErrorRate;
% 数据符号解调
% 将接收到的数据符号转换为列向量
equalizedSymbols = reshape(rxDataSymbols, [], 1);  
 

% 解调接收数据符号
receivedBits = pskdemod(equalizedSymbols, modulationOrder, pi/4);

% 误码率计算
errors = errorRate(originalBits, receivedBits);


fprintf('\nSymbol error rate = %d from %d errors in %d symbols\n',errors);