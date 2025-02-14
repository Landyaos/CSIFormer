snrValues = (0:3:30)';

ser_ls_mmse = [
    1.5188;
    1.2478;
    0.9884;
    0.7326;
    0.4986;
    0.2763;
    0.1002;
    0.0285;
    0.0077;
    0.0012;
    0
];

ser_mmse_mmse = [
    1.4383;
    1.1947;
    0.9547;
    0.7154;
    0.4883;
    0.2729;
    0.0994;
    0.0280;
    0.0075;
    0.0012;
    0
];

ser_csiEncoder_mmse = [
    1.3584;   % SNR=0 dB
    1.1299;   % SNR=3 dB
    0.8541;   % SNR=6 dB
    0.6391;   % SNR=9 dB
    0.4001;   % SNR=12 dB
    0.2046;   % SNR=15 dB
    0.0557;   % SNR=18 dB
    0.0154;   % SNR=21 dB
    0.0042;   % SNR=24 dB
    0.0010;   % SNR=27 dB
    0
];

ser_csiFormer_mmse = [
    1.3039;   % SNR=0 dB
    1.0496;   % SNR=3 dB
    0.8085;   % SNR=6 dB
    0.5927;   % SNR=9 dB
    0.3856;   % SNR=12 dB
    0.1887;   % SNR=15 dB
    0.0557;   % SNR=18 dB
    0.0144;   % SNR=21 dB
    0.0046;   % SNR=24 dB
    0.0006;   % SNR=27 dB
    0
];

ser_ideal_mmse = [
    1.2293;
    0.9900;
    0.7597;
    0.5468;
    0.3509;
    0.1690;
    0.0480;
    0.0111;
    0.0032;
    0.0005;
    0
];

% 生成表格
T = table(snrValues, ser_ls_mmse, ser_mmse_mmse, ser_csiEncoder_mmse, ser_csiFormer_mmse, ser_ideal_mmse);
T.Properties.VariableNames = {'SNR_dB', 'LS', 'L_MMSE', 'CSIEncoder', 'CSIFormer', 'Ideal'};

% 显示表格
disp(T);

% 假设你已经生成并保存了表格 T
% T 的变量名为：'SNR_dB', 'LS', 'L_MMSE', 'CSIEncoder', 'CSIFormer', 'Ideal'

% 使用 semilogy 绘制对数坐标图
figure;
hold on;

semilogy(T.SNR_dB, T.LS,        '-o', 'LineWidth', 1, 'DisplayName', 'LS');
semilogy(T.SNR_dB, T.L_MMSE,    '-s', 'LineWidth', 1, 'DisplayName', '(L)MMSE');
semilogy(T.SNR_dB, T.CSIEncoder, '-*', 'LineWidth', 1, 'DisplayName', 'CSIEncoder');
semilogy(T.SNR_dB, T.CSIFormer,  '-p', 'LineWidth', 1, 'DisplayName', 'CSIFormer');
semilogy(T.SNR_dB, T.Ideal,     '-d', 'LineWidth', 1, 'DisplayName', 'Ideal');

grid on;
xlabel('SNR (dB)');
ylabel('SER');
title('SER vs. SNR for Different Channel Estimation Algorithms');
legend('Location','best');
hold off;
% 假设 T 是之前生成的表格数据
T = table(snrValues, ser_ls_mmse, ser_mmse_mmse, ser_csiEncoder_mmse, ser_csiFormer_mmse, ser_ideal_mmse);
T.Properties.VariableNames = {'SNR_dB', 'LS', 'L_MMSE', 'CSIEncoder', 'CSIFormer', 'Ideal'};

% 创建一个图窗来显示表格
f = figure('Name','SER 数据表','Position',[100 100 800 300]);

% 创建 uitable 并将表格 T 显示出来
t = uitable('Parent', f, 'Data', T{:,:}, ...
    'ColumnName', T.Properties.VariableNames, ...
    'RowName', [], ...
    'Units', 'Normalized', 'Position',[0 0 1 1]);

% 如果需要将此图窗保存为图像，可以使用 exportgraphics
exportgraphics(f, 'SER_DataTable.png');
