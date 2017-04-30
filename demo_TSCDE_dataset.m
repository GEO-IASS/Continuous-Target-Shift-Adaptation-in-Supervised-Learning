clear all; clc

DATA=importdata('kin8fm.data');
X1=DATA'; 
X=X1(1:8,:);
Y=X1(9,:);
ntrain=300; ntest=300;

ind=randperm(600);
xtrain=X(ind(1:length(ind)/2));
xtest=X(ind(length(ind)/2 +1:end));

ytrain=Y(ind(1:length(ind)/2));
ytest=Y(ind(length(ind)/2 +1:end));


%normalization
xscale=std(xtrain,0);
yscale=std(ytrain,0);
xmean=mean(xtrain);
ymean=mean(ytrain);
xtrain_normalized=(xtrain-xmean)./repmat(xscale,[1 ntrain]);
ytrain_normalized=(ytrain-ymean)./repmat(yscale,[1 ntrain]);
xtest_normalized=(xtest-xmean)./repmat(xscale,[1 ntest]);
ytest_normalized=(ytest-ymean)./repmat(yscale,[1 ntest]);
xtest2=xtest;ytest2=ytest;MSE=[];
fp=fopen('MSE.txt','w');
for ITER=1:5
disp('Iteration Number') ;
ITER
%%%%%%%%%%%%%%%%%%%%%%%%% Estimating weights
 w=TSCDE(xtrain_normalized,ytrain_normalized,...
          xtest_normalized,ytest_normalized);
%w=2*pdf_Gaussian(ytrain,2.5,0.5);
%%%%%%%%%%%%%%%%%%%%%%%%% Estimating conditional density
xtest0=[4 6];
ytest0=linspace(-1,4,300);
xtest1=repmat(xtest0,length(ytest0),1);
ytest1=repmat(ytest0',1,length(xtest0));
xtest=xtest1(:)';
ytest=ytest1(:)';
ntest=length(xtest);
xtest_normalized=(xtest-xmean)./repmat(xscale,[1 ntest]);
ytest_normalized=(ytest-ymean)./repmat(yscale,[1 ntest]);
axis_limit=[2 10 0 4];
  
 %True conditional density for artificial data
 ptest=pdf_Gaussian(ytest,2,0.5);

[ph,ph_error,MSE1]=LSCDE(xtrain_normalized,ytrain_normalized,...
         xtest_normalized,ytest_normalized,w);
 MSE=[MSE MSE1];
 
fprintf(fp,'%f\n',MSE1);
     
 end
fclose(fp);
disp('Avg MSE = ')
mean(MSE)
disp('Std Dev of MSE = ')
std(MSE)

