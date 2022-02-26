# python-deep-learning-class-example
Python Deep Learning Class Example

## Back-propagation

### Output layer loss

$$
loss_{out} = (A_{out}-T)\cdot A_{out}\cdot (1-A_{out})
$$

### Hidden layer loss

$$
loss_{h} = (loss_{h+1} \cdot (W_{h+1})^{T})\cdot A_{h} \cdot (1-A_h)
$$

### Learning

$$
W_i = W_i - \alpha ((A_{i-1})^T \cdot loss_i
$$

$$
b_i = b_i - \alpha \cdot loss_i
$$

## LINK

[Google Colab](https://colab.research.google.com/drive/12-rHPTIDMai_jxTSlu62bMwvNCHAYOrb?usp=sharing)
