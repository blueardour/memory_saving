# copy from https://github.com/blueardour/model-quantization

import os, sys
import torch

def read_policy(filename, section='init', debug=False, verbose=print):
    if not os.path.isfile(filename):
        verbose("file no exist: %s" % filename)
        return []

    policies = []
    attr = None
    valid = False
    found = False  # found policy for the section
    with open(filename) as f:
        while(True):
            line = f.readline()
            if len(line) == 0:
                break

            items = line.strip('\n').strip(' ')
            if len(items) == 0 or items[0] == "#":
                continue
            items = items.split('#')
            items = items[0]

            items = items.split(':')
            if debug:
                verbose(items)
            if len(items) < 2:
                break

            if 'on' in items[0].split():
                if section in items[0].split():
                    policies = [{"trigger": [ int(x) for x in items[1].split() ] }]
                    attr = None
                    valid = False
                    found = True
                    continue
                else:
                    found = False

            if not found:
                continue

            if 'by_' in items[0]:
                if attr is None:
                    attr = dict()
                elif valid:
                    policies.append(attr)
                    valid = False
                    attr = dict()

            if attr is None:
                continue

            items[0] = items[0].strip()
            items[1] = items[1].strip()
            if ',' in items[1]:
                items[1] = items[1].split(',')
            if isinstance(items[1], list):
                items[1] = [ i.strip() for i in items[1]]
            elif ' ' in items[1]:
                items[1] = items[1].split(' ')

            for i, t in enumerate(items[1]):
                if t in ['True', 'true']:
                    items[1][i] = True
                elif t in ['False', 'false']:
                    items[1][i] = False

            attr[items[0]] = items[1]
            if 'by_' not in items[0]:
                valid = True

        if attr is not None and valid:
            policies.append(attr)

    return policies

def deploy_on_init(model, filename, verbose=print):
    if not hasattr(model, 'modules'):
        return

    tags_count = { 'conv': 0, 'fc': 0, 'eltwise': 0, 'shuffle': 0, 'concat': 0, 'norm': 0,
                   'gelu': 0, 'layernorm': 0, 'matmul': 0, 'softmax': 0 }
    for k in tags_count:
        # assign index
        verbose("assign index for module with tag {} ".format(k))
        for m in model.modules():
            if hasattr(m, 'update_quantization_parameter') and getattr(m , 'tag', '') == k:
                m.update_quantization_parameter(index=tags_count[k])
                tags_count[k] += 1

        # update config on init
        policies = read_policy(filename, k, verbose=verbose)
        verbose("loading '{}' section of policy".format(k))
        verbose(policies)
        for p in policies:
            attributes = p
            assert isinstance(attributes, dict), "Error attributes"
            for m in model.modules():
                if hasattr(m, 'update_quantization_parameter') and getattr(m , 'tag', '') == k:
                    m.update_quantization_parameter(**attributes)

def deploy_on_epoch(model, policies, epoch, optimizer=None, verbose=print):
    if not hasattr(model, 'modules'):
        return

    if len(policies) < 1:
        return

    assert 'trigger' in policies[0], "No trigger provided"
    feedbacks = []
    if epoch in policies[0]['trigger']:
        for p in policies:
            attributes = p
            assert isinstance(attributes, dict), "Error attributes"
            for m in model.modules():
                if hasattr(m, 'update_quantization_parameter'):
                    feedback = m.update_quantization_parameter(**attributes)
                    feedbacks.append(feedback)

    if optimizer is not None:
        assert isinstance(optimizer, torch.optim.SGD), 'reset_momentum is only supported on SGD optimizer currently'
        with torch.no_grad():
            for fd in feedbacks:
                if 'reset_momentum_list' in fd and isinstance(fd['reset_momentum_list'], list):
                    for p in fd['reset_momentum_list']:
                        param_state = optimizer.state[p]
                        if 'momentum_buffer' in param_state:
                            buf = param_state['momentum_buffer']
                            buf.mul_(0)
                            verbose("reset the momentum_buffer for tensor with id: %s" % id(p))

def deploy_on_iteration(model, policies, iteration, optimizer=None, verbose=print):
    deploy_on_epoch(model, policies, iteration, optimizer, verbose)

if __name__ == "__main__":
    print("Loading policy")
    policies = read_policy('config/srresnet-policy.txt', debug=True)
    print(policies)

