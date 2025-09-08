function [net,Errortest,YTest,YPred]=motores_1(X,data_mat)


    data_mat=normalize(data_mat);
    X1(1).imf=vmd(data_mat(:,1),'NumIMFs',X(5),'PenaltyFactor',X(6));
    % X1(2).imf=vmd(data_mat(:,2),'NumIMFs',X(5),'PenaltyFactor',X(6));
    % X1(3).imf=vmd(data_mat(:,3),'NumIMFs',X(5),'PenaltyFactor',X(6));
    % X1(4).imf=vmd(data_mat(:,4),'NumIMFs',X(5),'PenaltyFactor',X(6));
    % X1(5).imf=vmd(data_mat(:,5),'NumIMFs',X(5),'PenaltyFactor',X(6));
    % data_mat1=[X1(1).imf,X1(2).imf,X1(3).imf,X1(4).imf,X1(5).imf];
    data_mat1=X1(1).imf;

data_mat_tr=data_mat1(1:ceil(4/5*size(data_mat1,1)),:);
data_mat_ts=data_mat1(ceil(4/5*size(data_mat1,1))+1:end,:);

YTrain_tr = (data_mat_tr(:,end))';
XTrain_tr =  (data_mat_tr(:,1:end))';

XTrain_tr = num2cell(XTrain_tr,1);
YTrain_tr = num2cell(YTrain_tr,1);

YTest = (data_mat_ts(:,end))';
XTest =  (data_mat_ts(:,1:end))';

XTest = num2cell(XTest,1);
YTest = num2cell(YTest,1);

%% Define Network Architecture
% Step2: Create a LSTM network
numResponses = size(YTrain_tr{1},1);
% numResponses = size(YTrain_tr,1);
 featureDimension = size(XTrain_tr{1},1);
% featureDimension = size(data_mat,2);

numHiddenUnits = X(1);%100

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(500)  %%50
    dropoutLayer(0.1)  %%0.5
    fullyConnectedLayer(numResponses)
    regressionLayer];
%% Train option
maxepochs = X(2);%1000
InitialLearnRate = X(3);%0.005
miniBatchSize = 1;

options = trainingOptions('adam', ...  %%adam
    'MaxEpochs',maxepochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',InitialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',50, ...
    'LearnRateDropFactor',X(4), ...
    'ExecutionEnvironment','cpu',...
    'Verbose',0, ...
    'Plots','none');
%% Train the Network
 net = trainNetwork(XTrain_tr,YTrain_tr,layers,options);
 % YTrain_tr=cell2mat(YTrain_tr);
 % for ii=1:
 % net = trainnet(data_mat,YTrain_tr,layers,"mse",options);
 % net = trainNetwork(data_mat,YTrain_tr',layers,options);



net = resetState(net);
YPred = predict(net,XTest);

Errortest=rmse(cell2mat(YPred),cell2mat(YTest'));
RMSE=rmse(cell2mat(YPred),cell2mat(YTest'))
MAE=mae(cell2mat(YPred)-cell2mat(YTest'))
pre_MAPE = abs((cell2mat(YTest')-cell2mat(YPred))./cell2mat(YPred));
MAPE = mean(pre_MAPE(isfinite(pre_MAPE)))
C=0.5*( mse(cell2mat(YPred) - cell2mat(YTest')))+2*( mae(cell2mat(YPred) - cell2mat(YTest')))
NRMSE=norm((cell2mat(YPred) - cell2mat(YTest'))) / norm((cell2mat(YPred)-mean(cell2mat(YPred))))
%Errortest=mae(cell2mat(YPred)-cell2mat(YTest'));
%pre_MAPE = abs((cell2mat(YTest')-cell2mat(YPred))./cell2mat(YPred));
%Errortest = mean(pre_MAPE(isfinite(pre_MAPE)));
%Errortest =0.5*( mse(cell2mat(YPred) - cell2mat(YTest')))+2*( mae(cell2mat(YPred) - cell2mat(YTest')));
%Errortest = norm((cell2mat(YPred) - cell2mat(YTest'))) / norm((cell2mat(YPred)-mean(cell2mat(YPred))));
%t=X(2);
%figure
% plot(MAE,'*-b')
% hold on
% plot(RMSE,'*-g')
% hold on
% plot(MAPE,'*-r')
% hold on
% plot(C,'*-y')
% hold on
% plot(NRMSE,'*-k')

% figure
 YPred = (cell2mat(YPred(1:end, 1:end)));  %have to transpose as plot plots columns
%  plot(y1,'*-b')
% hold on
%  y2 = (cell2mat(YTest(1:end, 1:end))');
%  plot(y2,'.-k')
end
% MAE=mae((YPred)-(YTest'))
% pre_MAPE = abs(((YTest')-(YPred))./(YPred));
% MAPE = mean(pre_MAPE(isfinite(pre_MAPE)))
% C=0.5*( mse((YPred) - (YTest')))+2*( mae((YPred) - (YTest')))
% NRMSE=norm(((YPred) - (YTest'))) / norm(((YPred)-mean((YPred))))
% 
% plot(YPred,'*-b')
% hold on
% plot(YTest','.-k')