## Update

**2021-08-02**

1. Change quantization method

   ```python
   # update shift and clip val with exponential moving average
   def update_clip_val_shift(input, clip_val, shift, iteration, ema_decay):
       max_value, _ = torch.max(input, dim=0)
       min_value, _ = torch.min(input, dim=0)
       clip_range = max_value - min_value
       if iteration == 0:
           clip_val.data = clip_range
           shift.data = min_value
       else:
           clip_val.sub_((1 - ema_decay) * (clip_val - clip_range))
           shift.sub_((1 - ema_decay) * (shift - min_value))
       iteration.add_(1)
   
   # in forward():
   y = (x - shift) / clip_val * (level - 1)
   y = torch.round(y + noise)
   y = torch.clamp(y, min=0, max=level - 1)
   if training:
       save_for_backward(y, signed=False)
   y = y / (level - 1) * clip_val + shift
   ```

2. This version achieves 65.05% top-1 acc on CIFAR100, better than baseline DeiT-Ti (64.83%).

3. Add per head quantization

4. Add stochastic round. To use it, 

   ```yaml
   stochastic_round: True
   ```



**2021-07-27**

1. fix index error: matmul grad_clip2

2. fix gradient error: gradient of clip_val should be -1 if x < -clip_value

3. add initialization method for clip_value: MMSE, ACIQ, Entropy, code adopted from [Outlier Channel Splitting](https://github.com/cornell-zhang/dnn-quant-ocs)

   Clip value initialization

   ```python
   # main.py
   ms.policy.deploy_on_init(model, args.ms_policy, verbose=verbose, override_verbose=True)
   verbose(f"verbose model: {model}")
   
   model.to(device)
   
   initialize_clip_value(data_loader_val, model, device, verbose=verbose)	
   ```

   config

   ```yaml
   on fc:
     by_index: all
     enable: True
     level: 256
     requires_grad: True
     init_choice: entropy
   ```

   

