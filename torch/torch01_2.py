# gpu
# criterion # 56번째 줄

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # loss 정의 방법

##### GPU #####
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch: ', torch.__version__, '사용DEVICE: ', DEVICE)
# torch:  1.11.0+cu113 사용DEVICE:  cuda
# model, data 부분에 gpu넣어주면(명시) 된다.

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # GPU로 돌리겠다!!!
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

print(x, y) # tensor([1., 2., 3.]) tensor([1., 2., 3.]) # 스칼라가 3개 
print(x.shape, y.shape) # torch.Size([3]) torch.Size([3]) -> torch.Size([3, 1]) torch.Size([3, 1])

# 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim=1))
model = nn.Linear(1, 1).to(DEVICE) # -> GPU로 쓰겠다!!!
              # x의 1(앞), y의 1(뒤)


#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam')

criterion = nn.MSELoss() # 평가지표의 표준
# 인스턴스(개체, 객체) / 클래스(대문자)

#optimizer = optim.Adam(model.parameters(), lr = 0.01) # model.parameters()은 어떤 모델을 엮을것인지 즉 model = nn.Linear
optimizer = optim.SGD(model.parameters(), lr = 0.01) 

#print(optimizer)

# model.fit(x ,y , epochs=10000, batch_size=1)

def train(model, criterion, optimizer, x, y):
    #model.train()  # 훈련모드
    optimizer.zero_grad() # 기울기 초기화
    
    hypothesis = model(x) # x를 넣었을 때의 값이 hypothesis에 담김 # y = wx + b
    
    #loss = criterion(hypothesis, y) # 예측값과 실제값 비교 # MSE
    #loss = nn.MSELoss(hypothesis, y) # 에러
    #loss = nn.MSELoss()(hypothesis, y) # 정상작동 (방법1)
    loss = F.mse_loss(hypothesis, y) # 정상작동 (방법2)
    
    # 여기까지가 순전파
    
    loss.backward() # 기울기값 계산까지
    optimizer.step() # 가중치 수정(역전파)
    return loss.item() 

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epoch, loss))

print("==========================================")

#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval() # 훈련없이 평가만 하려고 함(평가모드)
    
    with torch.no_grad(): # grad 갱신하지 않겠음 / 1번만 돌리면 됨
        predict = model(x) 
        loss2 = criterion(predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss: ', loss2)

# result = model.predict([4])
result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값 : ', result.item())

'''
최종 loss:  0.0007705340976826847
4의 예측값 :  4.055666923522949
'''