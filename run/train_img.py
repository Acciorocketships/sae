from sae.sae_model import AutoEncoder
from sae.loss import get_loss_idxs, correlation
from torch import nn
import torch
import torchvision.transforms as transforms
import torchvision
import tqdm
import wandb

wandb_project = "sae-img"
model_path = "saved/sae_img.pt"
device = "cpu" if not torch.cuda.is_available() else "cuda"

class CNNAE(nn.Module):
    def __init__(self,
                 hidden_channels : int = 32,
                 latent_dim : int = 384,
                 act_fn : object = nn.Mish):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(hidden_channels, 2*hidden_channels, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*hidden_channels, 2*hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*hidden_channels, 2*hidden_channels, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(hidden_channels * 2 * 4 * 4, latent_dim)
        )

        self.decoder = nn.Sequential(
            # Linear
            nn.Linear(latent_dim, hidden_channels * 2 * 4 * 4),
            act_fn(),
            # Shape
            nn.Unflatten(1, (2*hidden_channels, 4, 4)),
            # CNN
            nn.ConvTranspose2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(hidden_channels, 3, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ImgSetAE(nn.Module):

    def __init__(self, cnn_dim=384, sae_dim=2048, max_n=16):
        super().__init__()
        self.cnn_ae = CNNAE(latent_dim=cnn_dim)
        self.set_ae = AutoEncoder(dim=cnn_dim, hidden_dim=sae_dim, max_n=max_n)


    def forward(self, x, b_idx, detach=True):
        cnn_feat = self.cnn_ae.encoder(x)
        img = self.cnn_ae.decoder(cnn_feat)
        if detach:
            cnn_feat = cnn_feat.detach()
        z = self.set_ae.encoder(cnn_feat, b_idx)
        set_feat, batch = self.set_ae.decoder(z)
        imgs = self.cnn_ae.decoder(set_feat)
        return (imgs, batch), img


    def encode(self, x, b_idx):
        # Shape [B*t,f]
        feat = self.cnn_ae.encoder(x).flatten(1)
        z = self.set_ae.encoder(feat, b_idx)
        return z

    def decode(self, z):
        feat, batch = self.set_ae.decoder(z)
        img = self.cnn_ae.decoder(feat.reshape(-1, 96, 2, 2))
        return img, batch


def main():
    if wandb_project is not None:
        wandb.init(project=wandb_project, group="sae")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    seq_len_range = [0,8]
    seq_lens = torch.arange(seq_len_range[0], seq_len_range[1]+1)
    b_idx = torch.repeat_interleave(torch.arange(seq_lens.numel(), device=device), seq_lens)
    val_seq_lens = torch.tensor([0,2,4,6,8])
    val_b_idx = torch.repeat_interleave(torch.arange(val_seq_lens.numel(), device=device), val_seq_lens)

    batch_size = seq_lens.sum().item()
    val_batch_size = val_seq_lens.sum().item()

    trainset = torchvision.datasets.CIFAR10(root='/tmp/datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/tmp/datasets', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size,
                                             shuffle=False, num_workers=0)
    testloader = iter(testloader)
    
    model = ImgSetAE(cnn_dim=384, sae_dim=128*seq_len_range[1], max_n=seq_len_range[1]).to(device)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.functional.mse_loss
    end2end = True

    # train loop
    epochs = 200
    pbar = tqdm.trange(epochs, position=0, desc="All Epochs", unit="epoch")
    loss = 0
    val_loss = 0
    it = 1
    for epoch in pbar:
        pbar2 = tqdm.tqdm(trainloader, position=1, desc=f"Epoch {epoch}", unit="batch", leave=False)
        for data in pbar2:
            targets, _ = data
            if targets.shape[0] != batch_size:
                continue
            targets = targets.to(device)
            opt.zero_grad()
            (recon, recon_b_idx), cnn_pred = model(targets, b_idx, detach=(not end2end))
            if not end2end:
                loss_data = model.set_ae.loss()
                size_loss = loss_data["size_loss"]
                set_loss = loss_data["mse_loss"]
                corr = loss_data["corr"]
                x_var = loss_data["x_var"]
                xr_var = loss_data["xr_var"]
                cnn_loss = loss_fn(cnn_pred, targets)
                loss = loss_data["loss"] + cnn_loss
            else:
                var = model.set_ae.get_vars()
                n_true = var["n"]
                n_pred = var["n_pred"]
                n_pred_logits = var["n_pred_logits"]
                pred_idx, tgt_idx = get_loss_idxs(n_pred, n_true)
                x = targets[var["perm"]]
                x = x[tgt_idx]
                recon = recon[pred_idx]
                set_loss = loss_fn(x, recon)
                size_loss = loss_fn(n_pred_logits, n_true.unsqueeze(-1).float().detach())
                cnn_loss = loss_fn(cnn_pred, targets)
                corr = correlation(x.flatten(1), recon.flatten(1))
                x_var = 0
                xr_var = 0
                # x_var = x.flatten(1).var(dim=0).mean()
                # xr_var = recon.flatten(1).var(dim=0).mean()
                loss = size_loss + cnn_loss + set_loss


            pbar2.set_description(f"train loss: {loss:.3f}")
            loss.backward()
            opt.step()
            if wandb_project is not None:
                wandb.log(
                    {"loss": loss, "set loss": set_loss, "size loss": size_loss, "cnn loss": cnn_loss, "corr": corr, "x_var": x_var, "xr_var": xr_var}, commit=True,
                )
            it += 1

        # Validation
        val_targets, _ = next(testloader)

        with torch.no_grad():
            (val_recon, val_recon_b_idx), val_cnn_pred = model(val_targets, val_b_idx)

        val_loss_data = model.set_ae.loss()
        val_size_loss = val_loss_data["size_loss"]
        val_set_loss = val_loss_data["mse_loss"]
        val_cnn_loss = loss_fn(val_cnn_pred, val_targets)
        if val_recon.shape[0] == val_targets.shape[0]:
            perm = model.set_ae.get_vars()["perm"]
            _, inv_perm = torch.sort(perm)
            val_recon = val_recon[inv_perm]


        b = val_batch_size
        vis = torchvision.utils.make_grid(torch.cat([val_targets[:b], val_cnn_pred[:b].clamp(-1, 1), val_recon[:b]], dim=0), nrow=b)

        pbar.set_description(f"val loss: {val_loss:.3f}")
        if wandb_project is not None:
            wandb.log(
                {
                    "val loss": val_loss,
                    "val set loss": val_set_loss,
                    "val cnn loss": val_cnn_loss,
                    "val image": wandb.Image(vis / 2 + 0.5),
                },
            commit=False)
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, model_path)


if __name__ == '__main__':
    main()