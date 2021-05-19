
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.autograd.profiler as profiler

import memory_saving as ms

import copy
import sys
import gc
import pickle
import pdb

def demo():
    seed = 2809
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019

    class Model(nn.Module):
        def __init__(self, inplace=False):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
            self.relu = ms.ReLU(inplace=inplace)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    model = Model()

    x = torch.rand(1, 64, 5, 5)
    x = x - 0.5

    y = model(x)
    z = y.sum()
    #z.backward()

    pdb.set_trace()

def test(iteration=10, inplace=False):
    seed = 2809
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019

    relu = ms.ReLU

    class Model(nn.Module):
        def __init__(self, inplace=False):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
            self.relu = relu(inplace=inplace)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    model = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ms.GELU(),
            #Flatten(),
            #ms.Linear(64 * 56 * 56, 100),
            ms.LayerNorm([64,56,56]),
            #ms.BatchNorm2d(64),
            #ms.Conv2d(64, 64, 3, bias=False),
            #nn.ReLU(inplace),
            #ms.cc.Conv2d(64, 64, 3, bias=False),
            #nn.ReLU(inplace),
            #ms.cc.Conv2d(64, 64, 3, bias=False),
            #nn.BatchNorm2d(64),
            #ms.ReLU(inplace),
            #ms.Conv2d(64, 64, 3, bias=False),
            #ms.ReLU(inplace),
            )
    #model = Model(inplace)
    #model = model.cuda()

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
        #x = x.cuda()
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
    seed = 23
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019

    frac = 1.0
    if torch.__version__ >= "1.8":
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory * frac
    print("total_memory {} GB".format(total_memory / 1024 / 1024))
    inplace = True
    ReLU = ms.ReLU
    model = nn.Sequential(
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),

            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
            ms.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            ReLU(inplace),
        )
    model = model.cuda()
    model.train()
    #model.eval()

    ##########
    def b2mb(x): return int(x/2**20)
    class TorchTracemalloc():
        def __init__(self, i):
            self.id = i

        def __enter__(self):
            self.begin = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
            return self
    
        def __exit__(self, *exc):
            self.end  = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
            self.used   = b2mb(self.end-self.begin)
            self.peaked = b2mb(self.peak-self.begin)
            print(f"{self.id:4d} delta used/peak {self.used:4d}/{self.peaked:4d}")

    buffer_list = {}
    def tensor_id(init=False, buffers=None):
        bl = buffer_list if buffers is None else buffers
        torch.cuda.empty_cache()
        gc.collect()
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
                        #bl[id(obj)] = obj.element_size(), obj.nelement()
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

    def run(time=2, debug=[1]):
        #print(model)
        torch.cuda.empty_cache()
        for i in range(time):
            with TorchTracemalloc(i) as tt:
                if i in debug:
                    tensor_id()  # only parameters 
                x = torch.rand(256,64,56,56)
                x = x.cuda()
                if i in debug:
                    print("x id {}\n".format(id(x)))
                    tensor_id() # x is obsersed

                if i in debug and torch.__version__ >= "1.6":
                    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
                        with profiler.record_function("model_inference"):
                            y = model(x)
                else:
                    y = model(x)

                z = y.sum()
                if i in debug:
                    print("y id {}, z id {}\n".format(id(y), id(z)))
                    tensor_id() # x, y, z are observed

                z.backward()
                if i in debug:
                    tensor_id() # x, y, z are observed

                x, y, z = None, None, None

            if i in debug and torch.__version__ >= "1.6":
                #print(prof.key_averages().table())
                pass

    ###########

    ## 0. 
    tensor_id(True)
    #for m in model.modules():
    #    _add_memory_hooks(m)

    ## 1. original pytorch module
    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = False
    run()

    #return

    ## 2. fuse conv and bn
    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = True
        if hasattr(m, 'level'):
            m.level = 256
    run()

    #return

    ## 3. quant saved tensor
    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = True
        if hasattr(m, 'level'):
            m.level = 255
    run()

    ## 4. quant saved tensor
    for m in model.modules():
        if hasattr(m, 'memory_saving'):
            m.memory_saving = True
        if hasattr(m, 'level'):
            m.level = 15
    run()

    ### 1. original pytorch module
    #for m in model.modules():
    #    if hasattr(m, 'memory_saving'):
    #        m.memory_saving = False
    #run()

    ##return

    ### 2. fuse conv and bn
    #for m in model.modules():
    #    if hasattr(m, 'memory_saving'):
    #        m.memory_saving = True
    #    if hasattr(m, 'level'):
    #        m.level = 256
    #run()

    ##return

    ### 3. quant saved tensor
    #for m in model.modules():
    #    if hasattr(m, 'memory_saving'):
    #        m.memory_saving = True
    #    if hasattr(m, 'level'):
    #        m.level = 255
    #run()

if __name__ == "__main__":
    #demo()
    test(inplace=True)
    #test(inplace=False)
    #profile()


