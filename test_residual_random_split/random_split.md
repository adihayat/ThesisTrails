
# Random residual split

Hi!<br>
see results below on adversary experiment with random split:<br>
comparing:
```
test_residual_split/derived1/train.log -> spliting conv1_1 based on TPM scores 
(32 kernels are splitted 13 in "fast" parittion  and 19 in "slow" partition with residual addition)
(RED curve)

### compared to 

test_residual_random_split/derived1/train.log -> spliting conv1_1 randomlly
(32 kernels are splitted 16 in "fast" partition , i.e. as is , 16 in "slow" partition i.e. with residual addition (GREEN)curve

```


The blue curve is the baseline.<br>
- NOTE: the residual TPM based split and the random split have been trained with the same hyper params


```python
%run ~/caffe/utils/plot_loss.py  /home/or/caffe/exp/test_residual_random_split/baseline/baseline_relu_lsuv.log   /home/or/caffe/exp/test_residual_random_split/derived1/train.log ~/or10/caffe/exp/test_residual_split/derived1/train.log  --avg 3
```


![png](output_2_0.png)


In this experiment, running the random split underperformed the baseline.<br>
while the TPM based split outperformed the baseline


