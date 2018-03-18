clear all
close all
load('costMatrix.mat')
h = homographyCostMat;
len = length(h);
for i = 1:len
    for j = 1:len
        if h(i,j)>30
            %h(i,j) = -1;
        end
    end
end
imagesc(costMatrix)
colorbar
figure
imagesc(h)
colorbar