"""Opt-in exact-pass building blocks. No loader, optimizer or launch authority."""
from contextlib import nullcontext
from propose_v2_exact_passes import schedule, verify
from seek.schemas import digest
from seek.three_way.lexical_pair import need


class BoundWindows:
    """Bind schedule indices to immutable ordered record IDs and corpus identities."""
    def __init__(self, records, annotations, corpus_hash, annotations_hash, world, seed=271028, accumulation=8):
        need(digest(records) == corpus_hash, 'corpus_binding')
        need(digest(annotations) == annotations_hash, 'annotation_binding')
        self.ids = tuple(r['id'] for r in records)
        need(bool(self.ids) and all(isinstance(i, str) for i in self.ids) and
             len(set(self.ids)) == len(self.ids), 'unique_record_ids')
        self.world = world
        self.windows = schedule(len(records), world, seed=seed, accumulation=accumulation)
        self.summary = verify(self.windows, len(records), world)
        self.binding = dict(corpus_hash=corpus_hash, annotations_hash=annotations_hash,
                            ordered_ids_hash=digest(self.ids), schedule_hash=digest(self.windows),
                            world_size=world, seed=seed, epochs=2, accumulation=accumulation)

    def rank_batches(self, rank):
        need(type(rank) is int and 0 <= rank < self.world, 'rank')
        for window in self.windows:
            yield dict(index=window['index'], epoch=window['epoch'], valid_records=window['valid_records'],
                slots=tuple(None if step[rank] is None else (step[rank], self.ids[step[rank]])
                            for step in window['microsteps']))

    def fetch(self, dataset, slot):
        need(slot is not None, 'empty_slot_requires_explicit_path')
        index, record_id = slot
        need(self.ids[index] == record_id, 'slot_binding')
        row = dataset[index]
        need(row['id'] == record_id, 'dataset_order_changed')
        return row


def causal_example_means(logits, labels):
    """Next-token CE: each row normalized by its own shifted valid target count."""
    import torch
    import torch.nn.functional as F
    need(logits.ndim == 3 and labels.ndim == 2 and logits.shape[:2] == labels.shape,
         'logits_labels_shape')
    need(logits.shape[1] >= 2 and labels.dtype == torch.long, 'labels_contract')
    target = labels[:, 1:]
    counts = target.ne(-100).sum(dim=1)
    need(bool((counts > 0).all()), 'no_shifted_supervision')
    shifted = logits[:, :-1, :]
    # At least float32 loss arithmetic; retain float64 in mechanical tests.
    if shifted.dtype in (torch.float16, torch.bfloat16):
        shifted = shifted.float()
    losses = F.cross_entropy(shifted.reshape(-1, shifted.shape[-1]), target.reshape(-1),
                             ignore_index=-100, reduction='none').reshape_as(target)
    return losses.sum(dim=1) / counts


def backward_ddp_window(ddp, slots, valid_records, world, forward_loss):
    """Caller owns zero_grad/update. forward_loss must traverse ddp for empty slots too.

    This helper performs no optimization and installs no training entry. A real-model
    empty path is deliberately not supplied by this module.
    """
    need(bool(slots) and type(valid_records) is int and valid_records > 0, 'window')
    need(type(world) is int and world > 0, 'world')
    for j, slot in enumerate(slots):
        # no_sync must cover BOTH forward and backward.
        with ddp.no_sync() if j + 1 < len(slots) else nullcontext():
            loss = forward_loss(ddp, slot)
            need(loss.ndim == 0, 'scalar_example_loss')
            (loss * world / valid_records).backward()
