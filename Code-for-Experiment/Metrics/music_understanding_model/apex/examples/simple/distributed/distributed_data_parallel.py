import torch
import argparse
import os
from apex import amp
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if args.distributed:
    model = DistributedDataParallel(model)

loss_fn = torch.nn.MSELoss()
for t in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()

if args.local_rank == 0:
    print("final loss = ", loss)