
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import memory_saving as ms

import copy
import sys
import gc
import pickle

def test(iteration=10):
    seed = 2809
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019

    model = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ms.BatchNorm2d(64),
            ms.Conv2d(64, 64, 3, bias=False),
            nn.ReLU(),
            ms.cc.Conv2d(64, 64, 3, bias=False),
            nn.ReLU(),
            ms.cc.Conv2d(64, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ms.Conv2d(64, 64, 3, bias=False),
            #ms.ReLU(),
            )
    model = model.cuda()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    model1 = copy.deepcopy(model)
    #model1 = pickle.loads(pickle.dumps(model))

    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = False

    for m in model1.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = True
        if hasattr(m, 'level'):
            m.level = 256

    model.train()
    model1.train()
    print(model)
    print(model1)

    def verbose(model, model1):
        for k, v in model.named_parameters():
            if v is not None and hasattr(v, 'grad') and v.grad is not None:
                print("{}: val {}-{}; grad {}-{}".format(k, v.max(), v.min(), v.grad.max(), v.grad.min()))
            else:
                print("{}: val {}-{}".format(k, v.max(), v.min()))
    
        for k, v in list(model.state_dict().items()):
            if 'running' in k:
                print(k, v.max(), v.min())
    
        print(" -----  ")
        for k, v in model1.named_parameters():
            if v is not None and hasattr(v, 'grad') and v.grad is not None:
                print("{}: val {}-{}; grad {}-{}".format(k, v.max(), v.min(), v.grad.max(), v.grad.min()))
            else:
                print("{}: val {}-{}".format(k, v.max(), v.min()))
    
        for k, v in list(model1.state_dict().items()):
            if 'running' in k:
                print(k, v.max(), v.min())

    verbose(model, model1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #, momentum=0.9)
    optimizer.zero_grad()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01) #, momentum=0.9)
    optimizer1.zero_grad()
    for i in range(iteration):
        print("index: ", i)
        x = torch.rand(512,64,56,56)
        x = x - 0.5
        x = x.cuda()
        x1 = x.clone()

        y = model(x)
        z = y.sum()
        z.backward()
        optimizer.step()

        y1 = model1(x1)
        z1 = y1.sum()
        z1.backward()
        optimizer1.step()

        print('z diff: ', (z1 - z).item(), ', z ', z.item())
    verbose(model, model1)

def profile():
    frac = 1.0
    if torch.__version__ >= "1.8":
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory * frac
    print("total_memory {} GB".format(total_memory / 1024 / 1024))
    model = nn.Sequential(
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(True),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(True),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(True),
        )
    model = model.cuda()
    model.train()

    ##########
    def b2mb(x): return int(x/2**20)
    class TorchTracemalloc():
        def __enter__(self):
            self.begin = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
            return self
    
        def __exit__(self, *exc):
            self.end  = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
            self.used   = b2mb(self.end-self.begin)
            self.peaked = b2mb(self.peak-self.begin)
            print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

    buffer_list = {}
    def tensor_id(init=False, buffers=None):
        bl = buffer_list if buffers is None else buffers
        torch.cuda.empty_cache()
        objs = gc.get_objects()
        def lookup(obj):
            if isinstance(obj, (tuple, list)):
                for ob in obj:
                    lookup(ob)
            elif isinstance(obj, torch.Tensor):
                if init and id(obj) not in bl:
                    bl[id(obj)] = obj.element_size(), obj.nelement()
                    print('obj', id(obj), obj.element_size(), obj.nelement())
                else:
                    if id(obj) not in bl:
                        print('+ obj', id(obj), obj.element_size(), obj.nelement())
                        bl[id(obj)] = obj.element_size(), obj.nelement()
        lookup(objs)
        print('')

    def _add_memory_hooks(mod):
        def _generate_mem_hook(hook_type):
            def hook(self, *args):
                string = hook_type
                if hasattr(self, 'needs_input_grad'):
                    string += "needs_input_grad({})".format(self.needs_input_grad)
                if hasattr(self, 'saved_tensors'):
                    string += "length of saved_tensors({})".format(len(self.saved_tensors))
                print("hook: " + string)
            return hook
            
        mod.register_forward_pre_hook(_generate_mem_hook('pre'))
        mod.register_forward_hook(_generate_mem_hook('fwd'))
        mod.register_backward_hook(_generate_mem_hook('bwd'))

    def run(time=2):
        print(model)
        torch.cuda.empty_cache()
        for i in range(time):
            with TorchTracemalloc() as tt:
                if i > 0:
                    tensor_id()  # only parameters 
                x = torch.rand(512,64,56,56)
                x = x.cuda()
                if i > 0:
                    print("x id {}\n".format(id(x)))
                    tensor_id() # x is obsersed
                y = model(x)
                if i > 0:
                    print("y id {}\n".format(id(y)))
                    tensor_id() # x, y are observed. 
                z = y.sum()
                if i > 0:
                    print("z id {}\n".format(id(z)))
                    tensor_id() # x, y, z are observed
                z.backward()
                if i > 0:
                    tensor_id() # x, y, z are observed
                del x, y, z

    ###########

    ## 0. 
    tensor_id(True)
    #for m in model.modules():
    #    _add_memory_hooks(m)

    ## 1. original pytorch module
    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = False
    print(model)
    run()

    #return

    ## 2. fuse conv and bn
    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = True
        if hasattr(m, 'level'):
            m.level = 256
    print(model)
    run()

    return

    ## 3. quant saved tensor
    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = True
        if hasattr(m, 'level'):
            m.level = 255
    run()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    #test()
    profile()

