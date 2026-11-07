"""Cannistraci-Hebb training options used by this implementation."""

def cht_opts(parser):
     p = parser.add_argument_group('cht')
     p.add_argument('--mask', type=str)
     p.add_argument(
         '--fixed-topology',
         action='store_true',
         default=False,
         help='Keep a supplied sparse mask fixed; disable all remove/regrow updates.',
     )
     p.add_argument('--pruner_statedict', type=str)
     p.add_argument("--iterative_warmup_steps", type=int, default=0)
     # DST parameters
     p.add_argument("--dst_scheduler", action="store_true", default=False)
     #added by kangyou
     p.add_argument("--dst_end_epoch_fraction", type=float, default=0.75)
     p.add_argument("--dst_start_epoch", type=int, default=0)
     #end kangyou
     p.add_argument("--zeta", type=float, default=0.3)
     p.add_argument("--update_interval", type=int, default=1)
     p.add_argument("--sparsity", type=float, default=0.7)
     p.add_argument("--remove_method", type=str, default="weight_magnitude")
     p.add_argument("--regrow_method", type=str, default="random")
     p.add_argument("--init_mode", type=str, default="xavier")
     p.add_argument("--chain_removal", action="store_true", default=False)
     p.add_argument("--T_decay", type=str, default="no_decay")
     p.add_argument("--adaptive_zeta", action="store_true", default=False)

     # topology init
     p.add_argument("--WS", action="store_true", default=False)
     p.add_argument("--ws_beta", type=float, default=0.25)
     p.add_argument("--BRF", action="store_true", default=False)
     p.add_argument("--brf_r", type=float, default=0.25)
     p.add_argument("--degree_dist", type=str, default="uniform")


     p.add_argument("--itop", action="store_true", default=False)
     p.add_argument("--EM_S", action="store_true", default=False)
     p.add_argument("--factor", type=float, default=0.01)

     p.add_argument("--granet", action="store_true", default=False)
     p.add_argument("--granet_init_sparsity", type=float, default=0.9)
     p.add_argument("--granet_init_step", type=int, default=0)

     p.add_argument("--history_weights", action="store_true", default=False)
     p.add_argument("--new_history_weights", action="store_true", default=False)

     p.add_argument("--gmp", action="store_true", default=False)
     p.add_argument("--pruning_scheduler", type=str, default="none", choices=["none", "linear", "granet", "s_shape"])
     p.add_argument("--pruning_method", type=str, default="none", choices=["none", "ri", "weight_magnitude", "MEST"])
     p.add_argument("--sparsity_distribution", type=str, default="uniform")
     p.add_argument("--early_stop", action="store_true", default=False)
     p.add_argument("--pruning_T_end", type=float, default=None)

     p.add_argument("--start_T", type=float, default=1.0)
     p.add_argument("--end_T", type=float, default=1.0)
     p.add_argument("--k", type=float, default=6.0)
