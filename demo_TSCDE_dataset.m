clear all; clc

DATA=importdata('Dataset.txt');
X1=DATA'; 
X=X1(1:8,:);
Y=X1(9,:);
ntrain=length(DATA)/2;
ntest=length(DATA)/2;

ind=randperm(length(DATA));
xtrain=X(ind(1:length(DATA)/2));
xtest=X(ind(length(DATA)/2 +1:end));

ytrain=Y(ind(1:length(DATA)/2));
ytest=Y(ind(length(DATA)/2 +1:end));


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
for ITER=1:100

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
%  plot(xteset,ptest);
%  p_noise=pdf_Gaussian(noise,1,1.5);
%  plot(noise,p_noise,'*');


[ph,ph_error,MSE1]==LSCDE(xtrain_normalized,ytrain_normalized,...
         xtest_normalized,ytest_normalized,w);
 MSE=[MSE MSE1];
 ITER    
 
%dlmwrite('MSE.txt',MSE1,'delimiter','\t','precision','%.3f');
fprintf(fp,'%f\n',MSE1);
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NMSE %%%%%%%%%%%%%%
% sum_w=sum(w);sum_wtr=sum(w_ytr);
% mse=sum((w./sum_w)-(w_ytr/sum_wtr).^2)/length(w);
% NMSE=[NMSE mse];
 end
fclose(fp);

figure(1);clf;hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(ydisp,ptr_ydisp,'b-','LineWidth',2)
plot(ydisp,pte_ydisp,'k-','LineWidth',2)
plot(ydisp,w_ydisp,'r-','LineWidth',2)
plot(ytrain,w,'g*','LineWidth',2,'Color',[0 200 0]/255)
%plot(xtr,wh_xtr,'bo','LineWidth',1,'MarkerSize',8)
legend('p_{tr}(y)','p_{te}(y)','w(y)','w-hat(y)',2)
xlabel('y')


dataset=1;
figure(2)
clf
hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(xtrain,ytrain,'ko','LineWidth',1,'MarkerSize',6)
plot(xtest2,ytest2,'x','LineWidth',1,'MarkerSize',6)
xtest_unique=unique(xtest);
for xtest_index=1:length(xtest_unique)
  x=xtest_unique(xtest_index);   
    cdf_scale=(xtest0(2)-xtest0(1))*0.8/max(max(ptest(xtest==x)),max(ph(xtest==x)/yscale));
    plot(xtest(xtest==x)+ptest(xtest==x)*cdf_scale,...
         ytest(xtest==x),'b-','LineWidth',2);
  
  plot(xtest(xtest==x)+ph(xtest==x)*cdf_scale/yscale,...
       ytest(xtest==x),'r-','LineWidth',2);
end
legend('Training Sample','Test Sample','True','Estimated',5)
title('Conditional Density Estimation')
axis(axis_limit)
xlabel('x')
ylabel('y')


