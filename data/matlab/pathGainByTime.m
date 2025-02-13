clear;
clc;
%% 参数设置
% 系统参数配置
numSubc = 256;                                   % FFT 长度
numGuardBands = [16;15];                         % 左右保护带
numPilot = (numSubc-sum(numGuardBands)-1)/4;      % 每根天线的导频子载波
numTx = 2;                                       % 发射天线数量
numRx = 2;                                       % 接收天线数量
numSym = 14;                                     % 每帧 OFDM 符号数
cpLength = 72;                                   % 循环前缀长度

% 调制参数配置
M = 4;                                           % QPSK 调制（M=4）

% 信道模型配置
sampleRate = 15.36e6;                            % 采样率
pathDelays = [0, 30, 70, 90, 110, 190, 410] * 1e-9;  % 路径时延
averagePathGains = [0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8];  % 平均路径增益
maxDopplerShift = 100;                           % 最大多普勒频移

% 信号导频分布配置
validSubcIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)), numSubc/2+1);

% 导频子载波配置
pilotIndicesAnt1 = (numGuardBands(1)+1:4:numSubc-numGuardBands(2))';   % 天线1导频位置
pilotIndicesAnt2 = (numGuardBands(1)+2:4:numSubc-numGuardBands(2))';   % 天线2导频位置
pilotIndicesAnt2(end) = pilotIndicesAnt1(end)-1;
pilotIndicesAnt1 = pilotIndicesAnt1(pilotIndicesAnt1~=numSubc/2+1);    % 去除 DC 子载波
pilotIndicesAnt2 = pilotIndicesAnt2(pilotIndicesAnt1~=numSubc/2+1);    % 去除 DC 子载波

% 构造 PilotCarrierIndices
pilotIndices = zeros(numPilot, numSym, numTx);
pilotIndices(:, :, 1) = repmat(pilotIndicesAnt1, 1, numSym); % 天线1
pilotIndices(:, :, 2) = repmat(pilotIndicesAnt2, 1, numSym); % 天线2

% 数据子载波配置
dataIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)), ...
                     [unique(pilotIndices); numSubc/2+1]);
numDataSubc = length(dataIndices);
numFrameSymbols = numDataSubc * numSym * numTx;

% OFDM 解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubc, ...
    'NumGuardBandCarriers', numGuardBands, ...
    'RemoveDCCarrier', true, ...
    'NumSymbols', numSym, ...
    'PilotOutputPort', true, ...
    'PilotCarrierIndices', pilotIndices, ...
    'CyclicPrefixLength', cpLength, ...
    'NumReceiveAntennas', numRx);

% OFDM 调制器
ofdmMod = comm.OFDMModulator('FFTLength', numSubc, ...
    'NumGuardBandCarriers', numGuardBands, ...
    'InsertDCNull', true, ...
    'NumSymbols', numSym, ...
    'PilotInputPort', true, ...
    'PilotCarrierIndices', pilotIndices, ...
    'CyclicPrefixLength', cpLength, ...
    'NumTransmitAntennas', numTx);

% 固定随机种子
minSeed = 0;
maxSeed = 2^32 - 1;
seed = randi([minSeed, maxSeed]);
disp(['Seed: ', num2str(seed)]);

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
    'PathGainsOutputPort', true, ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', seed);

mimoChannelInfo = info(mimoChannel);
pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
toffset = mimoChannelInfo.ChannelFilterDelay;

%% 多帧仿真，观察单个子载波的相位变化
snr = 20;             % 固定 SNR (dB)
numFrames = 100;      % 仿真帧数

% 选定观察对象
selected_subcarrier = floor(numDataSubc/2);  % 选取中间的数据子载波
selected_symbol = ceil(numSym/2);            % 中间的 OFDM 符号
selected_tx = 1;                             % 发射天线 1
selected_rx = 1;                             % 接收天线 1

% 预分配存储
hContinuous = zeros(numFrames,1);

for frame = 1:numFrames
    % 数据符号生成
    txSymStream = randi([0 M-1], numFrameSymbols, 1); 
    dataSignal = pskmod(txSymStream, M);
    dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
    
    % 固定导频
    pilotSignal = repmat(1+1i, numPilot, numSym, numTx);
    
    % OFDM 调制
    txSignal = ofdmMod(dataSignal, pilotSignal);
    
    % 信道传输
    [airSignal, pathGains] = mimoChannel(txSignal);
    airSignal = [airSignal((toffset+1):end,:); zeros(toffset,numRx)];
    
    % 加噪声
    rxSignal = awgn(airSignal, snr, "measured");
    
    % OFDM 解调
    [rxDataSignal, rxPilotSignal] = ofdmDemod(rxSignal);
    
    % 完美信道估计 (需要 ofdmChannelResponse 函数)
    % hPerfect 尺寸: [numDataSubc x numSym x numTx x numRx]
    hPerfect = ofdmChannelResponse(pathGains, pathFilters, ...
        numSubc, cpLength, dataIndices, toffset);
    
    % 记录指定子载波、OFDM符号、天线对的信道
    hContinuous(frame) = hPerfect(selected_subcarrier, selected_symbol, selected_tx, selected_rx);
end

%% 绘制幅度与相位（解包后）的变化
figure;
subplot(2,1,1);
plot(1:numFrames, abs(hContinuous), '-o');
xlabel('帧索引');
ylabel('幅度');
title('信道幅度随帧数变化');

% 对相位进行解包，以去除跨越 -π/π 的突变
unwrappedPhase = unwrap(angle(hContinuous));

subplot(2,1,2);
plot(1:numFrames, unwrappedPhase, '-o');
xlabel('帧索引');
ylabel('相位 (弧度, 解包后)');
title('信道相位随帧数变化（解包后）');
