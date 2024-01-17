import torch
from torch.optim import Adam
import wandb
import inspect
import traceback
from sae import get_loss_idxs, correlation
from sae import AutoEncoder
from sae import AutoEncoderPos
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_dspn import AutoEncoder as AutoEncoderDSPN
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN
from visualiser import Visualiser

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="saved/{name}-{dim}-{max_n}-{hidden_dim}.pt"

user = "" # wandb user name (for logging)
project = "sae-rand" # wandb project name

def experiments():
	# this dict specifies the experiments that will be run. the keys are the experiment names, and the values are lists of experiments
	# each experiment in the list must be provided with a model. the rest of the config will be drawn from the default config.
	# an experiment can also override a default config by setting (e.g. setting "save": True in the experiment dict)
	trials = {
		"sae": [{"model": AutoEncoder, "save": False}],
		"sae_pos": [{"model": AutoEncoderPos, "save": False}],
		# "dspn": [{"model": AutoEncoderDSPN}],
		# "rnn": [{"model": AutoEncoderRNN}],
		# "tspn": [{"model": AutoEncoderTSPN}],
	}
	default = {
		"dim": 16,
		"hidden_dim": 256,
		"max_n": 16,
		"epochs": 50000,
		"load": False,
		"save": False,
		"log": True,
		"runs": 1,
		"retries": 1,
	}
	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			config["model_path"] = model_path_base.format(name=name, dim=config["dim"], max_n=config["max_n"], hidden_dim=config["hidden_dim"])
			for run_num in range(config["runs"]):
				for retry in range(1,config["retries"]+1):
					try:
						run(**config)
						break
					except Exception as e:
						print("Trial {retry} failed:".format(retry=retry))
						print(traceback.format_exc())
						if retry < config["retries"]:
							print("Retrying...")


def run(
			dim = 6,
			hidden_dim = 64,
			max_n = 16,
			epochs = 100000,
			batch_size = 64,
			model_path = None,
			model = None,
			name = None,
			load = False,
			save = False,
			log = True,
			**kwargs,
		):

	if inspect.isclass(model):
		model = model(dim=dim, hidden_dim=hidden_dim, max_n=max_n, **kwargs)

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda:0"

	model = model.to(device=device)

	config = kwargs
	config.update({"dim": dim, "hidden_dim": hidden_dim, "max_n": max_n})

	if log:
		wandb.init(
			entity=user,
			project=project,
			group=name,
			config=config,
		)
		vis = Visualiser(visible=False)

	optim = Adam(model.parameters())

	if load:
		try:
			model_state_dict = torch.load(model_path, map_location=device)
			model.load_state_dict(model_state_dict)
		except Exception as e:
			print(e)

	for t in range(epochs):

		data_list = []
		size_list = []
		for i in range(batch_size):
			# n = torch.randint(low=1, high=max_n, size=(1,))
			# n = torch.tensor([max_n])
			n = torch.minimum(
				((max_n+1)**2 * torch.rand(1) / max_n).int(),
				 torch.tensor(max_n)
				)
			x = torch.randn(n[0], dim)
			data_list.append(x)
			size_list.append(n)
		x = torch.cat(data_list, dim=0)
		sizes = torch.cat(size_list, dim=0)
		batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

		x = x.to(device)
		batch = batch.to(device)

		xr, batchr = model(x, batch)

		loss_data = model.loss()
		loss = loss_data["loss"]

		loss.backward()
		optim.step()

		optim.zero_grad()

		# var = model.get_vars()
		# pred_idx, tgt_idx = get_loss_idxs(var["n_pred"], var["n"])
		# corr = correlation(var["x"][tgt_idx], var["xr"][pred_idx])

		if log:
			log_data = {
				**loss_data,
				# "corr": corr,
			}

			if t % 100 == 0:
				x0 = x[batch==0]
				xr0 = xr[batchr==0]
				vis.reset()
				vis.show_objects(xr0.cpu().detach())
				vis.show_objects(x0.cpu().detach(), alpha=0.3, linestyle="dashed")
				plt = vis.render()
				img = wandb.Image(plt)
				log_data["visualisation"] = img

			wandb.log(log_data)

	if save:
		try:
			model_state_dict = model.state_dict()
			torch.save(model_state_dict, model_path)
		except Exception as e:
			print(e)

	wandb.finish()



if __name__ == '__main__':
	experiments()
