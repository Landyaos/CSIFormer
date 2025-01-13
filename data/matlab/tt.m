% 假设你的信道矩阵 H 是 (numsubc, numsym)，导频矩阵 H_pilot 是 (numpilotsubc, numsym)
numsubc = 64;  % 子载波数
numsym = 14;   % 符号数
numpilotsubc = 6;  % 导频子载波数
numpilot = 6;   % 导频子载波个数

% 假设 H_pilot 是导频矩阵，包含导频信道信息
% 导频索引（子载波）的位置
pilot_indices = [1, 10, 20, 30, 40, 50];

% 模拟导频信道矩阵（numsubc x numsym）
H = rand(numsubc, numsym) + 1i * rand(numsubc, numsym);  % 假设的信道矩阵
H_pilot = H(pilot_indices, :);  % 从H中提取导频位置的信道数据

% 创建插值网格
[X, Y] = meshgrid(1:numsubc, 1:numsym);

% 创建导频数据的坐标
[pilot_X, pilot_Y] = meshgrid(pilot_indices, 1:numsym);

% 使用 griddata 对信道进行插值
H_interp = griddata(pilot_X(:), pilot_Y(:), H_pilot(:), X, Y, 'cubic');

% H_interp 即为插值后的信道矩阵
