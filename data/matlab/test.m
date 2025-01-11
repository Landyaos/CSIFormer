%% 参数设置
sampleRate = 1e6;            % 采样率
numTx = 2;                   % 发射天线数
numRx = 2;                   % 接收天线数
pathDelays = [0 1e-6];       % 路径时延示例
averagePathGains = [0 -3];   % 路径平均增益示例 (dB)
maxDopplerShift = 30;        % 最大多普勒频移 (Hz)
numCalls = 50;               % 连续调用次数 (时隙数)
txLen = 200;                 % 每次调用长度 (采样点数)

%% 创建 MIMO 信道对象
mimoChannel = comm.MIMOChannel( ...
    'SampleRate',                 sampleRate, ...
    'SpatialCorrelationSpecification', 'None', ...
    'PathDelays',                 pathDelays, ...
    'AveragePathGains',           averagePathGains, ...
    'MaximumDopplerShift',        maxDopplerShift, ...
    'NumTransmitAntennas',        numTx, ...
    'NumReceiveAntennas',         numRx, ...
    'FadingDistribution',         'Rayleigh', ...
    'PathGainsOutputPort',        true);

%% 准备测试信号 (示例：随机QPSK符号)
modOrder = 4;
txData = randi([0 modOrder-1], txLen, numTx);
txSignal = qammod(txData, modOrder);

%% 收集多次调用的 pathGains
% 每次调用返回 [txLen x P x numTx x numRx]
[~, pathCount, ~, ~] = size(mimoChannel(txSignal));
pathGainsRecord = zeros(txLen, pathCount, numTx, numRx, numCalls);

for k = 1:numCalls
    [~, pathGains] = mimoChannel(txSignal);
    pathGainsRecord(:,:,:,:,k) = pathGains;
end
% 选定: pathIndex, Tx=1, Rx=1
pathIndex = 1;
txAntenna = 1;
rxAntenna = 1;

% 提取对应维度: [N x numCalls]
%   - N = txLen (采样点数)
%   - numCalls (帧索引)
tmpGains = squeeze(pathGainsRecord(:, pathIndex, txAntenna, rxAntenna, :));
% tmpGains 大小 = [200 x 50]

% 将幅度转为 dB
tmpMagdB = 20*log10(abs(tmpGains));

% 构造坐标轴：X=帧索引(1~numCalls), Y=采样点(1~N)
[X, Y] = meshgrid(1:numCalls, 1:txLen);

figure;
imagesc(tmpMagdB);  % [N x numCalls] 的矩阵
set(gca, 'YDir', 'normal');  % 让 Y 轴从下到上
xlabel('帧索引 (numCalls)');
ylabel('采样点索引 (N)');
title('彩色图：单路径、单天线对');
colormap jet;       % 采用 jet 色带
colorbar;           % 显示色带刻度
