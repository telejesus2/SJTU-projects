J=0.02;
sigma=0.8;
R=0.06;
m=0.6;
g=9.8;

% boucle fermée
A=[0 1 0 0; 0 0 -g/(1+sigma) 0; 0 0 0 1;-m*g/J 0 0 0];
B=[0;0;0;1/J];
w=((m*g*g)/((1+sigma)*J))^(1/4);
p=[-w -2*w w*exp(3*i*pi/4) w*exp(-3*i*pi/4)];
K=place(A,B,p)

% boucle fermée hierarchisée
A2 = [0 1 0; 0 0 -g/(1+sigma); 0 0 0];
B2 = [0;0;1];
p2 = [-w w*exp(3*i*pi/4) w*exp(-3*i*pi/4)];
K2=place(A2,B2,p2)