clear all; clc
%Toy  Example
%Generating data
ntrain=300; ntest=300;
noise=1.5*randn(1,ntrain);
ytrain1=(1.5*randn(1,ntrain)+1);
ytrain2=(0.5*randn(1,ntrain)+2.5);
ytrain=0.4*ytrain1+0.6*ytrain2;
ytest=0.5*randn(1,ntest)+2.5;
xtrain=ytrain+3+noise;
xtest=ytest+3+1.5*randn(1,ntest);
%For display  
ydisp=linspace(-1,4.5,100);
ptr_ydisp=0.4.*(pdf_Gaussian(ydisp,1,1.5))+0.6.*(pdf_Gaussian(ydisp,2.5,0.5));
pte_ydisp=pdf_Gaussian(ydisp,2.5,0.5);
w_ydisp=pte_ydisp./ptr_ydisp;

ptr_ytr=0.4.*(pdf_Gaussian(ytrain,1,1.5))+0.6.*(pdf_Gaussian(ytrain,2.5,0.5));
pte_ytr=pdf_Gaussian(ytrain,2.5,0.5);
w_ytr=pte_ytr./ptr_ytr;

%normalization of data
xscale=std(xtrain,0);
yscale=std(ytrain,0);
xmean=mean(xtrain);
ymean=mean(ytrain);
xtrain_normalized=(xtrain-xmean)./repmat(xscale,[1 ntrain]);
ytrain_normalized=(ytrain-ymean)./repmat(yscale,[1 ntrain]);
xtest_normalized=(xtest-xmean)./repmat(xscale,[1 ntest]);
ytest_normalized=(ytest-ymean)./repmat(yscale,[1 ntest]);
xtest2=xtest;ytest2=ytest;MSE=[];
%%%%%%%%%%%%%%%%%%%%%%%%% Estimating weights%%%%%%%%%%%%%%
for ITER=1:1
 w=TSCDE(xtrain_normalized,ytrain_normalized,...
          xtest_normalized,ytest_normalized);

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
 ITER    
 end


figure(1);clf;hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(ydisp,ptr_ydisp,'b-','LineWidth',2)
plot(ydisp,pte_ydisp,'k-','LineWidth',2)
plot(ydisp,w_ydisp,'r-','LineWidth',2)
plot(ytrain,w,'g*','LineWidth',2,'Color',[0 200 0]/255)
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


