clear all
close all
load('costMatrix.mat')
h = costMatrix;
for i = 1:110
    for j = 1:110
        if h(i,j)>15
            h(i,j) = -1;
        end
    end
end
imagesc(costMatrix)
colorbar
figure
imagesc(h)
colorbar