## Update

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

   

