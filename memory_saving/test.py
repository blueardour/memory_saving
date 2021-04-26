
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import memory_saving as ms
import copy
import sys
import pickle

def test_layer(iteration=10):
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
            ms.ReLU(),
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

def profile_conv():
    #print("start precent / memory: %r, %r" % GPUInfo.gpu_usage())
    nn.Conv2d = custom_conv_bn
    #nn.ReLU = custom_relu
    model = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(True),
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

    def run(time=2):
        print(model)
        for i in range(time):
            with TorchTracemalloc() as tt:
                x = torch.rand(1024,64,56,56)
                x = x.cuda()
                y = model(x)
                z = y.sum()
                z.backward()

    run()

    model[0].memory_saving = True
    model[2].memory_saving = True
    model[4].memory_saving = True
    model[0].fm_intervals = 257
    model[2].fm_intervals = 257
    model[4].fm_intervals = 257
    run()

    model[0].fm_intervals = 255
    model[2].fm_intervals = 255
    model[4].fm_intervals = 255
    run()

    model[0].memory_saving = False
    model[2].memory_saving = False
    model[4].memory_saving = False
    model[1].memory_saving = True 
    model[3].memory_saving = True 
    model[5].memory_saving = True 
    run()

    model[0].memory_saving = True
    model[2].memory_saving = True
    model[4].memory_saving = True
    model[0].fm_intervals = 257
    model[2].fm_intervals = 257
    model[4].fm_intervals = 257
    run()

    model[0].fm_intervals = 255
    model[2].fm_intervals = 255
    model[4].fm_intervals = 255
    run()

if __name__ == "__main__":
    seed = 2809
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019
    test_layer()
    #profile_conv()

