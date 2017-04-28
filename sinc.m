function y=sinc(x)

y=ones(size(x));
i=(x~=0);
y(i)=sin(pi*x(i))./(pi*x(i));   
