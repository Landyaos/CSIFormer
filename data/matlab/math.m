% matlab 列存储优先
A = [1, 3, 5; 
     2, 4, 6];
disp(A(:)) 
[H_est,nVar] = nrChannelEstimate(carrier,rxGrid,refInd,refSym,'CDMLengths',cdmLen);
channelEstimate
nrChannelEstimateo
helpChannelEstimate