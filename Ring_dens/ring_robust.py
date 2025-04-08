import torch
from .policy import (
    AdaptiveThresholdPolicy,
    ConstantThresholdPolicy,
    ExponentialDecayThresholdPolicy,
    LinearThresholdPolicy,
    StepThresholdPolicy,
)
import torch.nn.functional as F

def create_negative_annealing_schedule(args):
    # different policies for choosing the outer threshold, which 
    # is initialized to the full circle
    if args.anneal_mode == "adaptive_anneal_on_loss":
        thres_outer_policy = AdaptiveThresholdPolicy(
            args.num_epoch,
            lower_is_better=False,
            init_thres=1.0, 
            min_thres=0.1, 
            max_thres=1.0, 
            delta=args.shrinkage,
            window=3,
        )
    elif args.anneal_mode == "linear_anneal":
        thres_outer_policy = LinearThresholdPolicy(
            args.num_epoch,
            30,
            init_thres=1.0, 
            min_thres=0.1, 
        )
    elif args.anneal_mode == "step_anneal":
        thres_outer_policy = StepThresholdPolicy(
            args.num_epoch,
            args.max_anneal_epoch,
            init_thres=1.0, 
            min_thres=0.1, 
        )
    elif args.anneal_mode == "exponential_decay_anneal":
        thres_outer_policy = ExponentialDecayThresholdPolicy(
            args.num_epoch,
            args.max_anneal_epoch,
            init_thres=1.0, 
            min_thres=0.1,
            decay=0.1,
        )
    else:
        thres_outer_policy = ConstantThresholdPolicy(
            init_thres=args.thres_outer,
        )

    # everyone uses constant inner threshold!
    thres_inner_policy = ConstantThresholdPolicy(
        init_thres=args.thres_inner,
    )

    return thres_outer_policy, thres_inner_policy

class Ring(object):

    def __init__(
            self,
            thres_outer=0.1,
            thres_inner=0.1,
            # if True, thres_inner is a fraction of the contents
            # in the outer ball
            dynamic_inner=True
        ):
        super().__init__()
        if not dynamic_inner: assert thres_inner < thres_outer
        self.thres_outer = thres_outer
        self.thres_inner = thres_inner
        self.dynamic_inner = dynamic_inner
    
    def compute_hard_num(self, neg_num):
        if self.dynamic_inner:
            outer_idx = round(self.thres_outer * neg_num)
            inner_idx = round(self.thres_inner * outer_idx)
        else:
            outer_idx = round(self.thres_outer * neg_num)
            inner_idx = round(self.thres_inner * neg_num)
        return outer_idx-inner_idx, outer_idx, inner_idx

    def get_negatives(self, src_emb, neg_emb, outer_idx, inner_idx):
        self.device = src_emb.device

        all_dps = torch.sum(src_emb[:, None, :] * neg_emb, dim=2)
        sorted_indices = torch.argsort(all_dps.flatten(), descending=True)

        ring_indices = sorted_indices[inner_idx:outer_idx]
        return ring_indices
