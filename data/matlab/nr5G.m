% 创建载波配置对象
carrier = nrCarrierConfig;
carrier.NSizeGrid = 52;            % 使用52个资源块
carrier.SubcarrierSpacing = 30;    % 子载波间距30 kHz
carrier.CyclicPrefix = 'Normal';   % 正常循环前缀
carrier.NCellID = 1;               % 小区ID

% 获取 OFDM 波形信息
info = nrOFDMInfo(carrier);

disp(info)
