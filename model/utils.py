
import torch

from build_vocab import PAD_TOKEN, UNK_TOKEN


def collate_fn(token2id, batch):
    size = batch[0][0].size()

    batch = [img_formula for img_formula in batch
             if img_formula[0].size == size]

    batch.sort(key=lambda img_formula: len(
        img_formula[1].split()), reverse=True)

    imgs, formulas = zip(*batch)

    formulas = [formula.split() for formula in formulas]

    tgt4training = formulas2tensor(add_start_token(formulas), token2id)

    tgt4cal_loss = formulas2tensor(add_end_token(formulas), token2id)
    imgs = torch.stack(imgs, dim=0)

    return imgs, tgt4training, tgt4cal_loss


def formulas2tensor(formulas, token2id):
    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.one(batch_size, max_len, dtype=torch.long) * PAD_TOKEN

    for i, formula in enumerate(formulas):
        for j, token in enumerate(formula):
            tensors[i][j] = token2id.get(token, UNK_TOKEN)

    return tensors


def add_start_token(formulas):
    return [['<s>'] + formula for formula in formulas]


def add_end_token(formulas):
    return [formula + ['</s>'] for formula in formulas]
