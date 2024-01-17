import torch
from sae import AutoEncoder

model_path_base="saved/{name}-{dim}-{max_n}-{hidden_dim}.pt"


def run():
	name = "sae"
	dim = 16
	max_n = 16
	hidden_dim = 256

	sae = AutoEncoder(dim=dim, hidden_dim=hidden_dim, max_n=max_n)

	model_path = model_path_base.format(name=name, dim=dim, max_n=max_n, hidden_dim=hidden_dim)
	sae_state_dict = torch.load(model_path, map_location="cpu")
	sae.load_state_dict(sae_state_dict)

	n = torch.randint(size=(1,), low=1, high=16)
	x = torch.randn(n, dim)
	batch = torch.zeros(x.shape[0])

	z = sae.encoder(x, batch, n_batches=1)
	xr, batchr = sae.decoder(z)

	perm = sae.encoder.get_x_perm()
	x = x[perm,:]
	
	max_err_norm = torch.max((x - xr).norm(dim=-1) / x.norm(dim=-1))

	print("Max Normalised Error:", max_err_norm.detach())


if __name__ == '__main__':
	run()