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
snr = 2;

sampleRate = 15.36e6;                                     % 采样率
pathDelays = [0 0.5e-6 1.2e-6];                           % 路径时延
averagePathGains = [0 -2 -5];                             % 平均路径增益
maxDopplerShift = 50;                                     % 最大多普勒频移

% 初始化有效子载波范围
validSubcarrierRange = (numGuardBands(1)+1):(numSubcarriers-numGuardBands(2));
numValidSubcarriers = length(validSubcarrierRange);

% 导频索引 (满足范围限制)
pilotIndices= [12 26 40 54; 13 27 41 55]; % 天线导频索引

% 随机 QPSK 导频符号
pilotQPSKSymbols = [1+1i, 1+1i, 1+1i, 1+1i];
pilotSymbols = pilotQPSKSymbols(randi(length(pilotQPSKSymbols), numPilots, numSymbols, numTransmitAntennas));

% 数据子载波数量
numDataSubcarriers = numSubcarriers-sum(numGuardBands)-numel(pilotIndices);
dataSymbolIndices = setdiff(validSubcarrierRange,unique(pilotIndices));

% 比特流生成
numBits = numDataSubcarriers * numSymbols * bitsPerSymbol;
bitStream = randi([0 1], numBits, 1);  % 随机生成比特流
% 调制成符号
dataSymbols = pskmod(bitStream, modulationOrder, pi/4, 'gray');  % 调制后的符号为复数形式
% 重塑数据符号为所需维度
dataSymbols = reshape(dataSymbols, numDataSubcarriers, numSymbols, numTransmitAntennas);

%% OFDM调制
% 1. 创建空的原始信号矩阵
originalSignal = zeros(numSubcarriers, numSymbols, numTransmitAntennas);

% 2. 填充数据符号和导频符号
originalSignal(dataSymbolIndices, :, :) = dataSymbols;
originalSignal(pilotIndices(1,:), :, 1) = 1 + 1i;  % 天线 1 的导频
originalSignal(pilotIndices(2,:), :, 2) = 1 + 1i;  % 天线 2 的导频

% 3. 对每个天线的信号进行 IFFT 操作
txSignal = ifft(originalSignal, numSubcarriers, 1);  % 在频域上对每列应用 IFFT

% 4. 添加循环前缀 (CP)
% 循环前缀添加的位置是原始信号的最后部分
txSignal = [txSignal(end - cpLength + 1:end, :, :); txSignal];  % 添加 CP

% 5. 重塑信号的维度，合并时间域信号和循环前缀
txSignal = reshape(txSignal, (numSubcarriers + cpLength) * numSymbols, numTransmitAntennas);

%% 信道模型
% MIMO信号模型初始化
channelModel = IdealMIMOChannel(...
    'NumTransmitAntennas', numTransmitAntennas, ...
    'NumReceiveAntennas', numReceiveAntennas, ...
    'NumSubcarriers', numSubcarriers, ...
    'NumSymbols', numSymbols);

% 通过信道模型获取接收信号和路径增益
[rxSignal, pathGains] = channelModel(txSignal); % pathGains: [总样本数, N_path, N_tx, N_rx]

% 初始化 CSI 矩阵
pathGains = squeeze(sum(pathGains, 2));
pathGains = reshape(pathGains, numSubcarriers+cpLength, numSymbols, numTransmitAntennas, numReceiveAntennas);
pathGains = pathGains(cpLength+1:end, :, :, :);
pathGains = fft(pathGains, numSubcarriers, 1);
H = permute(pathGains, [4,3,1,2]);
disp(size(H))
H_data = H(:,:,dataSymbolIndices,:);
% 噪声模型
awgnChannel = comm.AWGNChannel( ...
    'NoiseMethod', 'Signal to noise ratio (SNR)', ...
    'SNR', snr); % 设置 SNR 为 20 dB

% rxSignal = awgn(rxSignal, snr, 'measured');
% rxSignal = awgnChannel(rxSignal);

%% OFDM解调

rxSignal = reshape(rxSignal, numSubcarriers+cpLength, numSymbols, numTransmitAntennas);
rxSignal = rxSignal(cpLength+1:end,:,:);
rxSignal = fft(rxSignal, numSubcarriers, 1);
disp(rxSignal(pilotIndices(1,:),1,1))
rxDataSymbols = rxSignal(dataSymbolIndices, :, :);

%% 信道估计

% 初始化估计的 CSI 矩阵
estimatedCSI = zeros(numReceiveAntennas, numTransmitAntennas, numDataSubcarriers, numSymbols);
estimatedPilotCSI = zeros(numReceiveAntennas, numTransmitAntennas, numPilots, numSymbols);
% 基于导频符号的信道估计
for symIdx = 1:numSymbols
    for rx = 1:numReceiveAntennas
        for tx = 1:numTransmitAntennas
            % 提取接收导频符号
            receivedPilots = rxSignal(pilotIndices(tx, :), symIdx, rx);    % [numPilots, 1]
            transmitPilots = originalSignal(pilotIndices(tx, :), symIdx, tx);    % [numPilots, 1]
            % 估计导频子载波上的信道响应
            pilotCSI = receivedPilots ./ transmitPilots; % [numPilots, 1]
            estimatedPilotCSI(rx, tx, :, symIdx) = pilotCSI;
            % 插值到所有有效子载波
            estimatedCSI(rx, tx, :, symIdx) = interp1(pilotIndices(tx, :), ...
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
    for subcIdx = 1:numDataSubcarriers
        % 构造当前子载波的信道矩阵 H_sub (接收天线 x 发射天线)
        H_sub = squeeze(H_data(:, :, subcIdx, symIdx)); % [numReceiveAntennas, numTransmitAntennas]

        % 提取接收的符号向量 y_sub (接收天线 x 1)
        y_sub = squeeze(rxDataSymbols(subcIdx, symIdx, :)); % [numReceiveAntennas, 1]

        % 零强迫均衡: W = (H^H * H)^-1 * H^H
        W = pinv(H_sub); % [numTransmitAntennas, numReceiveAntennas]

        % 均衡后的符号: s = W * y
        equalizedSymbols(subcIdx, symIdx, :) = W * y_sub; % [numTransmitAntennas, 1]
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

