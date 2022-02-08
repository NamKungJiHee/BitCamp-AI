x = 10    # 내 맘대로 튜닝 가능
y = 10    # 목표값 
w = 1     #(weight 초기값)
lr = 0.2
epochs = 5

for i in range(epochs):
    predict = x * w    # x * w + b (machine이 생각하는 y의 값)
    loss = (predict - y) **2
    
    # 가중치와 epoch도 넣어서 아래 print를 수정
    print("Loss: ", round(loss, 4), "\tPredict: ", round(predict, 4), "\tepoch:", i + 1, "\tweight", round(w,4))
    
    ### 핑퐁 하려는 공식 ###
    up_predict = x * (w + lr) 
    up_loss = (y - up_predict) ** 2
        
    down_predict = x * (w - lr) 
    down_loss = (y - down_predict) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else: 
        w = w + lr
''' 
Loss:  0        Predict:  10    epoch: 1        weight 1
Loss:  4.0      Predict:  12.0  epoch: 2        weight 1.2
Loss:  0.0      Predict:  10.0  epoch: 3        weight 1.0
Loss:  4.0      Predict:  12.0  epoch: 4        weight 1.2
Loss:  0.0      Predict:  10.0  epoch: 5        weight 1.0
'''