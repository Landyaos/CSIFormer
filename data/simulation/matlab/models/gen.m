% 参数设置
N_subcarriers = 64;   % 子载波数
N_symbols = 14;       % 每个时隙的 OFDM 符号数
CP_length = 16;       % 循环前缀长度
N_tx = 2;             % 发射天线数
N_rx = 2;             % 接收天线数
SampleRate = 15.36e6; % 采样率
PathDelays = [0 1.5e-6 3e-6]; % 路径时延
AveragePathGains = [0 -3 -6]; % 平均路径增益（单位：dB）
MaxDopplerShift = 50;         % 最大多普勒频移

% 信号模型初始化
channelModel = comm.MIMOChannel(...
    'SampleRate', SampleRate, ...
    'NumTransmitAntennas', N_tx, ...
    'NumReceiveAntennas', N_rx, ...
    'PathDelays', PathDelays, ...
    'AveragePathGains', AveragePathGains, ...
    'MaximumDopplerShift', MaxDopplerShift, ...
    'FadingDistribution', 'Rayleigh', ...
    'NormalizePathGains', true, ... % 正则化路径增益
    'PathGainsOutputPort', true);   % 开启路径增益输出

% 数据符号和导频生成
modOrder = 4; % QPSK 调制阶数
dataSymbols = (randi([0 modOrder-1], N_subcarriers, N_symbols, N_tx) * 2 / modOrder - 1) + ...
              1j * (randi([0 modOrder-1], N_subcarriers, N_symbols, N_tx) * 2 / modOrder - 1);

% 导频插入 (基于子载波正交复用)
fixedPilot = (1 + 1j); % 固定导频值
pilotIndices = cell(N_tx, 1); % 每根天线的导频索引

% 设置天线导频位置
for t = 1:N_tx
    % 每隔 N_tx 子载波插入一个导频
    pilotIndices{t} = t:N_tx:N_subcarriers; 
end

% 导频分配：天线 t 的导频在其指定索引位置，其他位置填 0
for t = 1:N_tx
    for p = 1:N_tx
        if p == t
            dataSymbols(pilotIndices{t}, :, t) = fixedPilot; % 插入导频
        else
            dataSymbols(pilotIndices{t}, :, p) = 0; % 非本天线的导频位置填 0
        end
    end
end

% IFFT 变换：从频域到时间域
ofdmSymbols = ifft(dataSymbols, N_subcarriers, 1); % [N_subcarriers, N_symbols, N_tx]

% 添加循环前缀
ofdmSymbolsWithCP = [ofdmSymbols(end-CP_length+1:end, :, :); ofdmSymbols]; % [N_subcarriers + CP_length, N_symbols, N_tx]

% 序列化：准备输入信道
txSignal = reshape(ofdmSymbolsWithCP, [], N_tx); % [总样本数, N_tx]

% 通过信道模型获取接收信号和路径增益
[rxSignal, pathGains] = channelModel(txSignal); % pathGains: [总样本数, N_path, N_tx, N_rx]

% 初始化 CSI 矩阵
H = zeros(N_rx, N_tx, N_subcarriers, N_symbols); % [N_rx, N_tx, N_subcarriers, N_symbols]

% 计算 CSI
for rx = 1:N_rx
    for tx = 1:N_tx
        for k = 1:N_symbols
            % 提取当前符号范围的路径增益
            startIdx = (k-1) * (N_subcarriers + CP_length) + 1;
            endIdx = startIdx + (N_subcarriers + CP_length) - 1;
            symbolPathGains = pathGains(startIdx:endIdx, :, tx, rx); % [N_sample, N_path]
            
            % 聚合路径增益为时间域信道
            timeDomainH = sum(symbolPathGains, 2); % [N_sample, 1]
            
            % 去除 CP 并转换到频域
            timeDomainH_noCP = timeDomainH(CP_length+1:end); % 去除循环前缀
            freqDomainH = fft(timeDomainH_noCP, N_subcarriers); % [N_subcarriers, 1]
            
            % 存储 CSI
            H(rx, tx, :, k) = freqDomainH;
        end
    end
end

% 保存数据
save('MIMO_OFDM_Data.mat', 'dataSymbols', 'H', 'pilotIndices');
disp('数据和 CSI 提取完成，已保存为 MIMO_OFDM_Data.mat');

% 显示部分数据用于验证
disp('部分导频位置:');
disp(pilotIndices);
disp('部分 CSI 数据:');
disp(H(:, :, 1:5, 1)); % 显示前 5 个子载波的 CSI
