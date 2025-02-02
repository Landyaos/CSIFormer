clc; clear; close all;

% 参数
Nsubc = 64; % OFDM 子载波数
Nt = 2; % 发送天线数
Nr = 2; % 接收天线数
SNR_dB = 20; % 信噪比

% 生成随机正交导频信号 (Nt x Nsubc)
X = (randn(Nt, Nsubc) + 1j * randn(Nt, Nsubc)) / sqrt(2);

% 真实信道矩阵 H (Nr x Nt x Nsubc)
H_real = (randn(Nr, Nt, Nsubc) + 1j * randn(Nr, Nt, Nsubc)) / sqrt(2);

% 生成噪声 (Nr x Nsubc)
sigma_n = 10^(-SNR_dB/20);
N = sigma_n * (randn(Nr, Nsubc) + 1j * randn(Nr, Nsubc));

% 计算接收信号 Y = H * X + N
Y = zeros(Nr, Nsubc);
for k = 1:Nsubc
    Y(:,k) = H_real(:,:,k) * X(:,k);
end

% LS 估计 H_hat
H_hat = zeros(Nr, Nt, Nsubc);
for k = 1:Nsubc
    H_hat(:,:,k) = Y(:,k) * X(:,k)' * inv(X(:,k) * X(:,k)');
end

% 计算误差
error = norm(H_real - H_hat, 'fro') / norm(H_real, 'fro');
disp(['信道估计归一化误差: ', num2str(error)]);
