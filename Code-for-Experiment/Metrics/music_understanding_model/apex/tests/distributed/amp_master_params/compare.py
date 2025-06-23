import torch

model_params_rank0 = torch.load("rank0model.pth", map_location=lambda storage, loc: storage.cuda(0))
model_params_rank1 = torch.load("rank1model.pth", map_location=lambda storage, loc: storage.cuda(0))
master_params_rank0 = torch.load("rank0master.pth", map_location=lambda storage, loc: storage.cuda(0))
master_params_rank1 = torch.load("rank1master.pth", map_location=lambda storage, loc: storage.cuda(0))

for model_rank0, model_rank1, master_rank0, master_rank1 in zip(
    model_params_rank0,
    model_params_rank1,
    master_params_rank0,
    master_params_rank1
):
    assert torch.allclose(model_rank0, model_rank1), "Model param mismatch"
    assert torch.allclose(master_rank0, master_rank1), "Master param mismatch"
    assert torch.allclose(model_rank0, master_rank0.half(), rtol=.005), "Model-master mismatch"

print("OK:  Model and master params match across ranks.")