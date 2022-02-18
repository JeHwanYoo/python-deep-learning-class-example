# python-deep-learning-class-example
Python Deep Learning Class Example

## XOR Problem

![XOR Problem](https://i.ibb.co/jGscxQw/XOR.png)

## Examples

```python
xdata = np.array([[0, 0],[0, 1],[1, 0],[1, 1]]).reshape(4,2)
tdata = np.array([0, 1, 1, 0]).reshape(4,1)

xor_model = ANN([(2, 2), (2, 1)])
xor_model.train(xdata, tdata, epochs=30001)

test_data = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])

for x in test_data:
  y = xor_model.predict(x)
  if y >= 0.5:
    print(f'{x} = 1')
  else:
    print(f'{x} = 0')
```

(![image](https://user-images.githubusercontent.com/13535954/154632237-862ef94a-a2b6-40b6-a8f4-8ec35c9a5c58.png)

```plain
step =  0 loss value =  2.835243992618905
step =  3000 loss value =  2.6522861520968903
step =  6000 loss value =  2.1730020821222085
step =  9000 loss value =  1.7215880605383187
step =  12000 loss value =  0.6008394493810707
step =  15000 loss value =  0.24859044596202917
step =  18000 loss value =  0.14805128453811897
step =  21000 loss value =  0.10387565328910175
step =  24000 loss value =  0.0795236416934344
step =  27000 loss value =  0.06422180128812002
step =  30000 loss value =  0.05376022277921273
[0 0] = 0
[0 1] = 1
[1 0] = 1
[1 1] = 0
```

## LINK

[Blog](https://jehwanyoo.net/2022/02/18/%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D%EC%9D%84-%ED%95%B4%EB%B3%B4%EC%9E%90-4%EC%9E%A5-%EB%94%A5-%EB%9F%AC%EB%8B%9D/)

[Google Colab](https://colab.research.google.com/drive/1qal22C73QJZ8mIop1yFyZA5kyDJtuIjz?usp=sharing)
