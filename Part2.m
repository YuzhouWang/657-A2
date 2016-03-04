%% Load data

load('DataF');

%% Normalization

[r,c] = size(fea);

for i=1:c
    meanval = mean(fea(:,i));
    
    for j=1:r
        fea(j,i) = fea(j,i) - meanval;
    end
end

%% Apply PCA to reduce the dimension to be 4

% %one way
% covMatrix = cov(fea);
% [eigenVector, eigenValues] = eig(covMatrix);
% feaD4 = fea*eigenVector(:, end:-1:end-3);

%another way
[~,score,latent,~,~,~] = pca(fea);
feaD4 = score(:,1:4);

%% Cluster the data using agglomerative algorithms in hierarchical clustering 
%???trsform gnd from [0,9] to [1,10]
gndNew = ind2vec((gnd+1)');

%Single Link
% %one way
% tree_singlelink = linkage(feaD4, 'single');
% cluster_singlelink = cluster(tree_singlelink, 'maxclust', 10);

%another way
cluster_singlelink = clusterdata(feaD4, 'linkage', 'single', 'maxclust', 10);

%evaluate the clustering result in terms of Separation-Index, Rand-Index, and F-measure
output_singlelink = ind2vec(cluster_singlelink');
[c,cm,ind,per] = confusion(gndNew, output_singlelink);
correct =  sum(cluster_singlelink == (gnd+1));

%% find the separation index:

[r,c] = size(fea);


% Accuracy_singlelink = 
% Precision_singlelink =  
% Recall_singlelink = 
% Fmeasure_singlelink = 

%% Rand index:
[r,c] = size(fea);
M = (r*(r-1))/2;

a = 0; %number both same
b = 0; %number both different

for i = 1:r
    for j = i+1:r
        if (gnd(i) == gnd(j)) && (cluster_singlelink(i) == cluster_singlelink(j))
            a = a + 1;
        elseif (gnd(i) ~= gnd(j)) && (cluster_singlelink(i) ~= cluster_singlelink(j))
            b = b +1;
        end
    end
end

(a + b) / M
%




%%
%Complete Link
cluster_completelink = clusterdata(feaD4, 'linkage', 'complete', 'maxclust', 10);

%Ward's Algorithm(minimum variance algorithm)
cluster_ward = clusterdata(feaD4, 'linkage', 'ward', 'maxclust', 10);

output_ward = ind2vec(cluster_ward');
[c,cm,ind,per] = confusion(gndNew, output_ward);
correct2 =  sum(cluster_ward == (gnd+1));

%the number of clusters from 2 to 15
for i = 2:15
    cluster_ward = clusterdata(feaD4, 'linkage', 'ward', 'maxclust', i);
end

%% Rand index:
[r,c] = size(fea);
M = (r*(r-1))/2;

a = 0; %number both same
b = 0; %number both different

for i = 1:r
    for j = i+1:r
        if (gnd(i) == gnd(j)) && (cluster_completelink(i) == cluster_completelink(j))
            a = a + 1;
        elseif (gnd(i) ~= gnd(j)) && (cluster_completelink(i) ~= cluster_completelink(j))
            b = b +1;
        end
    end
end

(a + b) / M
%

%% f-measure
mij = zeros(10,10);
ni = zeros(10,1);
mj = zeros(10,1);

for i = 1:r
    mij(cluster_completelink(i), gnd(i)+1) = mij(cluster_completelink(i), gnd(i)+1) + 1;
    ni(cluster_completelink(i)) = ni(cluster_completelink(i)) + 1;
    mj(gnd(i)+1) = mj(gnd(i)+1) + 1;
end

for i=1:10
    for j=1:10
        precision(i,j) = mij(i,j) / ni(i);
        recall(i,j) = mij(i,j) / mj(j);
    end
end
%% Cluster the data using K-means algorithm

%the number of clusters from 2 to 15
for i = 2:15
    [cluster_kmeans,centroid] = kmeans(feaD4, i);
end
%% Cluster the data using Fuzzy C-means algorithm

%set the exponent for partition matrix to 2
options = [2; nan; nan ; nan];
[center, cluster_fcmeans, obj_fcn] = fcm(feaD4, 10, options);


%%
clusters = max( cluster_fcmeans);
classes(i) = cluster_fcmeans(:,i) = clusters(i);
%class1 = cluster_fcmeans(:,1) 
