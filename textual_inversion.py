import argparse
from copy import deepcopy
from pathlib import Path

import torch
from munch import munchify
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import get_loader
from solver.latent_diffusion import get_solver


def text_embedding_init(model, placeholder_text:str="*", init_text:str=""):
    # add placeholder text to tokenizer
    model.tokenizer.add_tokens([placeholder_text])

    # get token id of placeholder/init text
    iid = model.tokenizer.encode(init_text, add_special_tokens=False)[0]
    pid = model.tokenizer.convert_tokens_to_ids(placeholder_text)

    # extend embeddings if placeholder is newly added.
    model.text_encoder.resize_token_embeddings(len(model.tokenizer))

    # initialize placeholder token with init text embedding
    token_embeds = model.text_encoder.get_input_embeddings().weight.data
    token_embeds[pid] = token_embeds[iid].clone()

    # change CLIP embedding: need to replace by nn.Embedding, not Tensor.
    new_embed = torch.nn.Embedding(len(model.tokenizer),token_embeds.size(1)).to(model.device)
    new_embed.weight.data = token_embeds # copy data 
    model.text_encoder.text_model.embeddings.token_embedding = deepcopy(new_embed) # no clone for nn.Embedding

    model.text_encoder.requires_grad_(False)
    model.text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    return model

def textual_inversion(model, x0, t, text_emb):
    noise = torch.randn_like(x0)
    at = model.alpha(t)
    xt = at.sqrt() * x0 + (1-at).sqrt() * noise

    pred = model.unet(xt, t.squeeze(), encoder_hidden_states=text_emb)['sample']
    target = noise
    loss = F.mse_loss(pred, target)
    return loss

def sample(model, x0, prompt, NFE:int):
    with torch.no_grad():
        uc, c = model.get_text_embed("", prompt)

    times = reversed(torch.linspace(0, 1000, NFE+1).to(x0.device))
    xt = torch.randn_like(x0)

    pbar = tqdm(times[1:], desc='Sampling')
    for idx, t in enumerate(pbar):
        at = model.alpha(t.long())
        pred_uc, pred_c = model.predict_noise(xt, t, uc, c)
        pred = pred_uc + 7.5 * (pred_c - pred_uc)
        x0t = (xt - (1-at).sqrt()*pred)/at.sqrt()
        if idx < NFE - 1:
            atn = model.alpha(times[idx+2].long())
            xt = atn.sqrt()*x0t + (1-atn).sqrt()*pred
        else:
            xt = x0t
    return xt

def train(model, loader, num_epochs:int, placeholder:str, init:str, **kwargs):

    templates = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
    ]

    model = text_embedding_init(model, init_text=init, placeholder_text=placeholder)
    optim = torch.optim.AdamW(list(model.text_encoder.text_model.embeddings.token_embedding.parameters()),
                              lr=1e-4, weight_decay=1e-2)

    for epoch in range(num_epochs):
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for i, (img1, _) in enumerate(pbar):
            img1 = img1 * 2 - 1
            with torch.no_grad():
                x0 = model.encode(img1.cuda())

            t = torch.randint(0, 1000, (1,1,1,1)).cuda()

            template = templates[torch.randint(len(templates), (1,)).item()]
            text = template.format(placeholder)

            # text embedding
            txtid = model.tokenizer(text,
                                    padding='max_length',
                                    max_length=model.tokenizer.model_max_length,
                                    return_tensors='pt')
            text_emb = model.text_encoder(txtid.input_ids.cuda())[0]
            
            loss = textual_inversion(model, x0, t,
                                    text_emb=text_emb)
            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix({'loss': loss.item()})
         
            #### eval model ####
            if kwargs.get('eval_freq') is not None:
                if i % kwargs.get('eval_freq') == 0 or i == len(loader)-1: 
                    with torch.no_grad():
                        sample_x1 = sample(model, x0, f'a photo of a {placeholder}', NFE=50)
                        sample_x1 = model.decode(sample_x1)
                        save_image(torch.cat([img1, sample_x1.cpu()], dim=0),
                                kwargs.get('workdir').joinpath(f'evaluation/{kwargs.get('name')}/{epoch}_{i}.png'),
                                normalize=True,
                                scale_each=True)

            #### save text embedding ####
            if kwargs.get('save_freq') is not None:
                if i % kwargs.get('save_freq') == 0:
                    torch.save(model.text_encoder.get_input_embeddings(), 
                               kwargs.get('workdir').joinpath(f'{kwargs.get('name')}_{epoch}_{i}.pt'))

    return model.text_encoder.get_input_embeddings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, default=Path('embeddings'))
    parser.add_argument('--root', type=Path, help="path to domain images")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--placeholder', type=str, default='*')
    parser.add_argument('--init', type=str, default='a photo')
    parser.add_argument('--save_freq', type=int, default=1000, help="epochs to save")
    parser.add_argument('--eval_freq', type=int, default=1000, help="epochs to eval")
    parser.add_argument('--name', dtype=str, help="name of domain, e.g cat")
    args = parser.parse_args()

    args.workdir.joinpath('checkpoint').mkdir(exist_ok=True, parents=True)    
    args.workdir.joinpath('evaluation').mkdir(exist_ok=True, parents=True)    

    loader = get_loader(args.root, args.root, batch_size=1)
    config = munchify({'num_sampling': 1000})
    model = get_solver(name='ddim', solver_config=config, device='cuda', pipe_dtype=torch.float32)

    train(model=model,
          loader=loader,
          **vars(args))