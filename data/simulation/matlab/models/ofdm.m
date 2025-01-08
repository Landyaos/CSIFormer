% 参数设置
numSubcarriers = 64;         % FFT 长度
numGuardBands = [6;6];       % 左右保护带
numPilots = 4;               % 每根天线的导频子载波
numTransmitAntennas = 2;     % 发射天线数量
numReceiveAntennas = 2;      % 接收天线数量
numSymbols = 14;             % 每帧 OFDM 符号数
cpLength = 16;               % 循环前缀长度
modulationOrder = 4;         % QPSK 调制（M=4）
bitsPerSymbol = log2(modulationOrder);

numValidSubcarriers = numSubcarriers- sum(numGuardBands); % 有效子载波

%% OFDM调制
% 导频索引 (满足范围限制)
pilotIndicesAntenna1 = [7; 8; 9; 10]; % 天线 1 导频索引
pilotIndicesAntenna2 = [55; 56; 57; 58]; % 天线 2 导频索引
% 构造 PilotCarrierIndices (3D 矩阵, NPilot-by-NSym-by-NT)
pilotIndices = zeros(numPilots, numSymbols, numTransmitAntennas);
pilotIndices(:, :, 1) = repmat(pilotIndicesAntenna1, 1, numSymbols); % 天线 1
pilotIndices(:, :, 2) = repmat(pilotIndicesAntenna2, 1, numSymbols); % 天线 2

% 随机 QPSK 导频符号
pilotSymbols = ones(numPilots, numSymbols, numTransmitAntennas);
pilotSymbols(:,:,:) = 100;

% 数据子载波数量
numDataSubcarriers = numSubcarriers-sum(numGuardBands)-(numPilots*numTransmitAntennas);

% 数据符号
dataSymbols = ones(numDataSubcarriers, numSymbols, numTransmitAntennas);
for i = 1:numTransmitAntennas
    for j = 1:numSymbols
        dataSymbols(:,j,i)=(1+numDataSubcarriers*(j-1)):(numDataSubcarriers*j);
    end
end

% OFDM调制器
ofdmMod = comm.OFDMModulator('FFTLength', numSubcarriers, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'NumSymbols', numSymbols, ...
                             'InsertDCNull', false,...
                             'PilotInputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumTransmitAntennas', numTransmitAntennas);
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubcarriers, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'NumSymbols', numSymbols, ...
                             'RemoveDCCarrier', false,...
                             'PilotOutputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumReceiveAntennas', numReceiveAntennas);


% OFDM 调制
txSignal = ofdmMod(dataSymbols, pilotSymbols);

txSignal = reshape(txSignal, numSubcarriers+cpLength, numSymbols, numTransmitAntennas);
txSignal = txSignal(cpLength+1:end,:,:);
txSignal = fft(txSignal, [], 1);
txSignal = fftshift(txSignal,1);
% disp(txSignal(:,1,1))
a = ones(500,2);
disp(length(a))
