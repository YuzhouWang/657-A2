%% Load data

load('DataD');

%% Preprocess using Z-score Normalization

[r,c] = size(fea);
feaZ = zeros(r,c);

for i = 1:c
    meanVal = mean(fea(:,i));
    stdVal = std(fea(:,i));
    for j = 1:r
        feaZ(j,i) = (fea(j,i) - meanVal) / stdVal;
    end
end

% the first half of data for training, the second for testing 
split = r/2;
Train_data = feaZ(1:split,:);
Train_group = gnd(1:split,:);
Test_data = feaZ(split+1:end,:);
Test_group = gnd(split+1:end,:);


%% Clssify data using K-NN classifier

k = 1:2:31;
error_knn = zeros(1, length(k));
Accuracy_knn = zeros(1, length(k));
maxAccuracy_knn = 0;
Precision_knn = zeros(1, length(k));
Recall_knn = zeros(1, length(k));
Fmeasure_knn = zeros(1, length(k));

trTime = zeros(length(k),5);
clTime = zeros(length(k),5);

%Using 5-fold cross validation on the training set
indices = crossvalind('Kfold', split, 5);

%evaluate k-NN on the values k=[1, 3, 5, 7, ..., 31]
for i = 1:length(k)

    error = zeros(1,5);
    precision = zeros(1,5);
    recall = zeros(1,5);
    fmeasure = zeros(1,5);
    
    for j = 1:5
        crossTrain_data = Train_data((indices~=j),:);
        crossTrain_group = Train_group((indices~=j),:);
%         label_knn = knnclassify(Test_data, crossTrain_data, crossGroup, k(i), 'euclidean', 'nearest');
        tic;
        model_knn = fitcknn(crossTrain_data, crossTrain_group, 'NumNeighbors', k(i));
        trTime(i,j) = toc;
        
        tic;
        label_knn = predict(model_knn, Train_data((indices==j),:));
        clTime(i,j) = toc;
        
        error(j) =  sum(label_knn ~=  Train_group((indices==j),:));
%         tpfp = sum(label_knn == 1);
%         tpfn = sum(gnd(split+1:end,:)==1);
%         tp = sum(label_knn == (gnd(split+1:end,:) == 1));
%                 
%         precision(j) = tp/tpfp;
%         recall(j) = tp/tpfn;
%         fmeasure(j) = 2 / (1/precision(j) + 1/recall(j));
    end
    error_knn(i) = mean(error);
    Accuracy_knn(i) = (1-error_knn(i)/(r-split))*100;
%     Precision_knn(i) = mean(precision);
%     Recall_knn(i) = mean(recall);
%     Fmeasure_knn(i) = mean(fmeasure);

    if maxAccuracy_knn < Accuracy_knn(i)
        maxAccuracy_knn = Accuracy_knn(i);
        bestk = k(i);
    end
end

figure;
plot(k, Accuracy_knn);
xlabel('k');
ylabel('accuracy(%)');
print(gcf, 'images\K-NN', '-dpng', '-r0');

%% Now that we have the best k, train and test with all data:
error_knn = zeros(1, 20);
Accuracy_knn = zeros(1, 20);
maxAccuracy_knn = 0;
Precision_knn = zeros(1, 20);
Recall_knn = zeros(1, 20);
Fmeasure_knn = zeros(1, 20);

for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran= randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_group = gnd(ran(1:split),:);
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_group = gnd(ran(split+1:end),:);

    model_knn = fitcknn(randomTrain_data, randomTrain_group, 'NumNeighbors', bestk);

    label_knn = predict(model_knn, randomTest_data);

    
    error_knn(i) = sum(label_knn ~= randomTest_group);
    Accuary_knn(i) = (1-error_knn(i)/(r-split))*100;
    tpfp = sum(label_knn == 1);
    tpfn = sum(randomTest_group ==1);
    tp = sum(label_knn == (randomTest_group == 1));

    Precision_knn(i) = tp/tpfp;
    Recall_knn(i) = tp/tpfn;
    Fmeasure_knn(i) = 2 / (1/Precision_knn(i) + 1/Recall_knn(i));

end

%the average and standard deviation of classification performance
avgAccuracy_knn = mean(Accuracy_knn);
stdAccuracy_knn = std(Accuracy_knn);
avgPrecision_knn = mean(Precision_knn);
stdPrecision_knn = std(Precision_knn);
avgRecall_knn = mean(Recall_knn);
stdRecall_knn = std(Recall_knn);
avgFmeasure_knn = mean(Fmeasure_knn);
stdFmeasure_knn = std(Fmeasure_knn);

trainTime_knn = mean(mean(trTime));
classifyTime_knn = mean(mean(clTime));



%% Clssify data using RBF kernel SVM classifier

addpath 'libsvm-3.21/matlab'

c = [0.1, 0.5, 1, 2, 5, 10, 20, 50];
gamma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10];

error_svm = zeros(length(c), length(gamma));
accuracy_svm = zeros(length(c), length(gamma));
maxAccuracy_svm = 0;
precision_svm = zeros(length(c), length(gamma));
recall_svm = zeros(length(c), length(gamma));
fmeasure_svm = zeros(length(c), length(gamma));

traintime = zeros(length(c), length(gamma), 5);
classifytime = zeros(length(c), length(gamma), 5);

 indices = crossvalind('Kfold', split, 5);

%???every time the result is different
%select soft margin penalty term "c" from the set [0.1, 0.5, 1, 2, 5, 10, 20, 50] 
for i = 1:length(c)
    %select kernel width parameter "gamma" from the set [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    for j = 1:length(gamma)
        %Using 5-fold cross validation on the training set
       
        error = zeros(1,5);
        precision = zeros(1,5);
        recall = zeros(1,5);
        fmeasure = zeros(1,5);
%         Accuracy = zeros(1,5);
        for k = 1:5
            crossTrain_data = Train_data((indices~=k),:);
            crossTrain_group = Train_group((indices~=k),:);
            
% %             SVMStruct = svmtrain(crossTrain_data, crossTrain_group, 'kernel_function', 'rbf', 'boxconstraint', c(i), 'rbf_sigma', gamma(j));
% %             label_svm = svmclassify(SVMStruct, Test_data);
%             
%???not sure about KernelScale equals to gamma
            tic;
            SVMModel = fitcsvm(crossTrain_data, crossTrain_group, 'KernelFunction', 'rbf', 'BoxConstraint', c(i), 'KernelScale', gamma(j));
            traintime(i, j, k) = toc;
            
            tic;
            [label_svm, score_svm] = predict(SVMModel, Test_data); 
            classifytime(i, j, k) = toc;
            
            error(k) =  sum(label_svm ~= gnd(split+1:end,:));
            tpfp = sum(label_svm == 1);
            tpfn = sum(gnd(split+1:end,:)==1);
            tp = sum(label_svm == (gnd(split+1:end,:) == 1));

            precision(k) = tp/tpfp;
            recall(k) = tp/tpfn;
            fmeasure(k) = 2 / (1/precision(k) + 1/recall(k));
        end
        error_svm(i, j) = mean(error); 
        accuracy_svm(i, j) = (1-error_svm(i, j)/(r-split))*100;
        precision_svm(i, j) = mean(precision);
        recall_svm(i, j) = mean(recall);
        fmeasure_svm(i, j) = mean(fmeasure);

%         %using libsvm fuction 
%         for k = 1:5
%             crossTrain_data = Train_data((indices~=k),:);
%             crossTrain_group = Train_group((indices~=k),:);
%             %if set '-v 5', return accuracy of classification using 5-fold cross validation ranther than a svmmodel
%             model_svm = libsvmtrain(crossTrain_group, crossTrain_data, ['-t 2 -b 1 -c ',num2str(c(i)),' -g ',num2str(gamma(j))]);
%             [label_svm, accuracy, prob_estimates] = libsvmpredict(Test_group, Test_data, model_svm);
%             Accuracy = accuracy(1);
%         end
%           accuracy_svm(i, j) = mean(Accuracy);

        if (maxAccuracy_svm < accuracy_svm(i, j))
            maxAccuracy_svm = accuracy_svm(i, j);
            bestc = c(i);
            bestgamma = gamma(j);
            bestlabel_svm = label_svm;
            bestscore_svm = score_svm;
        end
    end
end

%???how to caculate a matrix's standard variance
Accuracy_svm = zeros(1,length(c)*length(gamma));
Precision_svm = zeros(1,length(c)*length(gamma));
Recall_svm = zeros(1,length(c)*length(gamma));
Fmeasure_svm = zeros(1,length(c)*length(gamma));

for k = 1:length(c)*length(gamma)
    for i = 1:length(c)
        for j = 1:length(gamma)
            Accuracy_svm(k) = accuracy_svm(i,j);
            Precision_svm(k) = precision_svm(i,j);
            Recall_svm(k) = recall_svm(i,j);
            Fmeasure_svm(k) = fmeasure_svm(i,j);            
        end
    end
end
%the average and standard deviation of classification performance
% avgAccuracy_svm = mean(mean(accuracy_svm));
avgAccuracy_svm = mean(Accuracy_svm);
stdAccuracy_svm = std(Accuracy_svm);
avgPrecision_svm = mean(Precision_svm);
stdPrecision_svm = std(Precision_svm);
avgRecall_svm = mean(Recall_svm);
stdRecall_svm = std(Recall_svm);
avgFmeasure_svm = mean(Fmeasure_svm);
stdFmeasure_svm = std(Fmeasure_svm);

trainTime_svm = mean(mean(mean(traintime)));
classifyTime_svm = mean(mean(mean(classifytime)));

% set class label from 1,-1 to 1,0
Test_group(Test_group == 1) = 1;
Test_group(Test_group == -1) = 0;

%???not sure about transform score_svm to [0,1]
for i = 1:r-split
    for j = 1:2
        bestscore_svm(i,j) = (bestscore_svm(i,j)-min(bestscore_svm(:,j))) / (max(bestscore_svm(:,j))-min(bestscore_svm(:,j)));
    end
end
%???not sure, if not transform, the curve is weird
prob = zeros(r-split, 2);
for i = 1:r-split
    for j = 1:2
        prob(i,j) = bestscore_svm(i,3-j);
    end
end

%plot ROC curve
[tpr,fpr,thresholds] = roc(Test_group',prob');
figure;
plotroc(Test_group',prob');

%% Clssify data using Naive Bayes classifier

error_naivebayes = zeros(1, 20);
Accuary_naivebayes = zeros(1, 20);
Precision_naivebayes = zeros(1, 20);
Recall_naivebayes = zeros(1, 20);
Fmeasure_naivebayes = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);

for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran= randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_group = gnd(ran(1:split),:);
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_group = gnd(ran(split+1:end),:);
    
    tic;
    model_naivebayes = fitNaiveBayes(randomTrain_data, randomTrain_group);
    trainTime(i) = toc;
    
    tic; 
    label_naivebayes = predict(model_naivebayes, randomTest_data);
    classifyTime(i) = toc;
    
    error_naivebayes(i) = sum(label_naivebayes ~= randomTest_group);
    Accuary_naivebayes(i) = (1-error_naivebayes(i)/(r-split))*100;
    tpfp = sum(label_naivebayes == 1);
    tpfn = sum(randomTest_group == 1);
    tp = sum(label_naivebayes == (randomTest_group == 1));

    Precision_naivebayes(i) = tp/tpfp;
    Recall_naivebayes(i) = tp/tpfn;
    Fmeasure_naivebayes(i) = 2 / (1/Precision_naivebayes(i) + 1/Recall_naivebayes(i));
end 

%the average and standard deviation of classification performance
avgAccuracy_naivebayes = mean(Accuary_naivebayes);
stdAccuracy_naivebayes = std(Accuary_naivebayes);
avgPrecision_naivebayes = mean(Precision_naivebayes);
stdPrecision_naivebayes = std(Precision_naivebayes);
avgRecall_naivebayes = mean(Recall_naivebayes);
stdRecall_naivebayes = std(Recall_naivebayes);
avgFmeasure_naivebayes = mean(Fmeasure_naivebayes);
stdFmeasure_naivebayes = std(Fmeasure_naivebayes);

trainTime_naivebayes = mean(trainTime);
classifyTime_naivebayes = mean(classifyTime);

%% Clssify data using Decision Tree classifier

error_decisiontree = zeros(1,20);
Accuary_decisiontree = zeros(1,20);
Precision_decisiontree = zeros(1, 20);
Recall_decisiontree = zeros(1, 20);
Fmeasure_decisiontree = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);
for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran = randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_group = gnd(ran(1:split),:);
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_group = gnd(ran(split+1:end),:);
    
    %fitctree function returns a classification tree and fitrtree function returns a regression tree
    tic;
    model_decisiontree = fitctree(randomTrain_data, randomTrain_group);
    trainTime(i) = toc;
    
    tic;
    [label_decisiontree,score_decisiontree,node,cnum] = predict(model_decisiontree, randomTest_data);
    classifyTime(i) = toc;
    
    error_decisiontree(i) = sum(label_decisiontree ~= randomTest_group);
    Accuary_decisiontree(i) = (1-error_decisiontree(i)/(r-split))*100;
    tpfp = sum(label_decisiontree == 1);
    tpfn = sum(randomTest_group ==1);
    tp = sum(label_decisiontree == (randomTest_group == 1));

    Precision_decisiontree(i) = tp/tpfp;
    Recall_decisiontree(i) = tp/tpfn;
    Fmeasure_decisiontree(i) = 2 / (1/Precision_decisiontree(i) + 1/Recall_decisiontree(i));
end 

%the average and standard deviation of classification performance
avgAccuracy_decisiontree = mean(Accuary_decisiontree);
stdAccuracy_decisiontree = std(Accuary_decisiontree);
avgPrecision_decisiontree = mean(Precision_decisiontree);
stdPrecision_decisiontree = std(Precision_decisiontree);
avgRecall_decisiontree = mean(Recall_decisiontree);
stdRecall_decisiontree = std(Recall_decisiontree);
avgFmeasure_decisiontree = mean(Fmeasure_decisiontree);
stdFmeasure_decisiontree = std(Fmeasure_decisiontree);

trainTime_decisiontree = mean(trainTime);
classifyTime_decisiontree = mean(classifyTime);

%% Clssify data using Neural Network classifier

error_neuralnetwork = zeros(1,20);
Accuary_neuralnetwork = zeros(1,20);
Precision_neuralnetwork = zeros(1, 20);
Recall_neuralnetwork = zeros(1, 20);
Fmeasure_neuralnetwork = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);
for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran= randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_group = gnd(ran(1:split),:)';
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_group = gnd(ran(split+1:end),:)';
    
%???trasform randomTrain_group and randomTest_group
    randomTrain_target = zeros(2,split);
    randomTest_target = zeros(2,r-split);
    
    for j = 1:split
        if randomTrain_group(1,j) == 1
            randomTrain_target(1,j) = 1;
            randomTrain_target(2,j) = 0;
        else if randomTrain_group(1,j) == -1
            randomTrain_target(1,j) = 0;
            randomTrain_target(2,j) = 1;
            end
        end 
    end
    
    for j = 1:split
        if randomTest_group(1,j) == 1
            randomTest_target(1,j) = 1;
            randomTest_target(2,j) = 0;
        else if randomTest_group(1,j) == -1
            randomTest_target(1,j) = 0;
            randomTest_target(2,j) = 1;
            end
        end 
    end
    
    %create a Neural Network
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize);
    
    %train the Neural Network
    tic;
    [net, trainrecord] = train(net, randomTrain_data', randomTrain_target);
    trainTime(i) = toc;

%     nntraintool;
%     view(model_neuralnetwork);

    %test the Neural Network
    tic;
    testOutput = net(randomTest_data');
    classifyTime(i) = toc;
    
    label_neuralnetwork = zeros(1,r-split);
    for k = 1:r-split
        if testOutput(k) > 0.5
            label_neuralnetwork(k) = 1;
        else
            label_neuralnetwork(k) = -1;
        end
    end
        
    error_neuralnetwork(i) = sum(label_neuralnetwork ~= randomTest_group);
    Accuary_neuralnetwork(i) = (1-error_neuralnetwork(i)/(r-split))*100;
    tpfp = sum(label_neuralnetwork == 1);
    tpfn = sum(randomTest_group ==1);
    tp = sum(label_neuralnetwork == (randomTest_group == 1));

    Precision_neuralnetwork(i) = tp/tpfp;
    Recall_neuralnetwork(i) = tp/tpfn;
    Fmeasure_neuralnetwork(i) = 2 / (1/Precision_neuralnetwork(i) + 1/Recall_neuralnetwork(i));
 
%     figure;
%     plotperform(trainrecord);
%     figure;
%     plottrainstate(trainrecord);
%     figure;
%     ploterrhist(errors);
%     figure;
%     plotconfusion(randomTest_group,label_neuralnetwork);
%     figure;
%     plotroc(randomTest_group,label_neuralnetwork);
end 

%the average and standard deviation of classification performance
avgAccuracy_neuralnetwork = mean(Accuary_neuralnetwork);
stdAccuracy_neuralnetwork = std(Accuary_neuralnetwork);
avgPrecision_neuralnetwork = mean(Precision_neuralnetwork);
stdPrecision_neuralnetwork = std(Precision_neuralnetwork);
avgRecall_neuralnetwork = mean(Recall_neuralnetwork);
stdRecall_neuralnetwork = std(Recall_neuralnetwork);
avgFmeasure_neuralnetwork = mean(Fmeasure_neuralnetwork);
stdFmeasure_neuralnetwork = std(Fmeasure_neuralnetwork);

trainTime_neuralnetwork = mean(trainTime);
classifyTime_neuralnetwork = mean(classifyTime);



