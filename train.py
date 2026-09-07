# train.py
"""Train a small CNN or MLP on Fashion-MNIST in JAX/Flax."""
import os

from optim.pns_eigenadam import profile_pns_eigenadam_curvature
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
from absl import app, flags

import jax
import jax.numpy as jnp
import sys
from jax import tree_util as jtu

print("DEBUG sys.executable:", sys.executable)
print("DEBUG jax version:", jax.__version__)
import jaxlib
print("DEBUG jaxlib version:", jaxlib.__version__)
print("DEBUG devices from train.py:", jax.devices())
print("DEBUG backend from train.py:", jax.default_backend())
from absl import app, flags


from flax.metrics.tensorboard import SummaryWriter

import optax

from data import get_datasets
from engine.flax_engine import create_train_state, make_train_step, make_eval_step, cross_entropy_loss
from optim.eigentools import (
    init_eigentracking,
    track_eigenstate,
    extract_momentum_pytree,
    extract_precond_directions,
    densify_precond_directions,
    _leaf_offsets_of,
    compute_soap_perlayer_sin2,
    compute_muon_topk_energy,
)
from optim.factory import build_curvature_matvec_fn, get_optimizer
from models.mlp import MLP
from models.resnet import SmallResNet, ResNet30, ResNet18
from models.vit_small import VisionTransformer
from utils import (
    load_config,
    maybe_make_dir,
    init_wandb,
    log_scalar_dict,
    get_exp_dir_path,
    save_loss_curves,
    _sanitize_name,
    init_eigen_tracking_csv,
    append_eigen_tracking_row,
    init_hessian_probe_csv,
    append_hessian_probe_row,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "config/config.yaml", "Path to config.yaml file.")
flags.DEFINE_string(
    "exp_name",
    None,
    "Override exp_name from the config for the output folder.",
)


def _make_effective_update_transform_fn(
    tx,
    params,
    opt_state,
    base_grads,
    *,
    base_mode: str,
    direction_scale: float,
    normalize_lr: bool,
    learning_rate: float,
):
    """Return transform(direction) -> effective update delta.

    Probes how the optimizer preconditions a given direction by comparing
    tx.update(base + scale*direction) against tx.update(base), normalised by
    direction_scale (and optionally lr). Identical logic to train_lm.py.
    """
    if base_mode == "zero":
        reference_grads = jtu.tree_map(jnp.zeros_like, base_grads)
    elif base_mode == "grad":
        reference_grads = base_grads
    else:
        raise ValueError(
            "effective_curvature_transform_base must be one of {'zero', 'grad'}."
        )

    reference_updates, _ = tx.update(reference_grads, opt_state, params)

    lr_scale = 1.0
    if normalize_lr:
        lr_scale = max(abs(float(learning_rate)), 1e-30)
    denom = float(direction_scale) * lr_scale

    def transform(direction):
        if base_mode == "zero":
            probe_grads = jtu.tree_map(lambda d: direction_scale * d, direction)
        else:
            probe_grads = jtu.tree_map(
                lambda g, d: g + direction_scale * d, base_grads, direction
            )
        probe_updates, _ = tx.update(probe_grads, opt_state, params)
        return jtu.tree_map(
            lambda probe, reference: (probe - reference) / denom,
            probe_updates,
            reference_updates,
        )

    return transform


def _make_probe_measurement_fn(model_def, tx, curvature_batch, batch_stats):
    """Build a JIT-compiled probe measurement function for image classification.

    Applies one optimizer step on the fixed curvature batch and measures
    loss/accuracy before and after, matching the LLM probe_update mode.
    """
    probe_images, probe_labels = curvature_batch

    def _fwd(params):
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats
            out, _ = model_def.apply(
                variables, probe_images, train=False, mutable=["batch_stats"]
            )
        else:
            out = model_def.apply(variables, probe_images, train=False)
        logits = out[0] if isinstance(out, tuple) else out
        loss = cross_entropy_loss(logits, probe_labels)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == probe_labels)
        return loss, acc

    @jax.jit
    def compute_probe_measurement(params, opt_state):
        (loss_before, acc_before), grads = jax.value_and_grad(
            _fwd, has_aux=True
        )(params)
        probe_updates, _ = tx.update(grads, opt_state, params)
        probe_params = optax.apply_updates(params, probe_updates)
        loss_after, acc_after = _fwd(probe_params)
        return {
            "probe_loss_before": loss_before,
            "probe_loss_after": loss_after,
            "probe_loss_reduction": loss_before - loss_after,
            "probe_acc_before": acc_before,
            "probe_acc_after": acc_after,
            "probe_acc_gain": (acc_after - acc_before) / jnp.maximum(
                1.0 - acc_before, 1e-8
            ),
        }

    return compute_probe_measurement


def _should_run_eigen_tracking_for_step(cfg, completed_step: int) -> bool:
    """Return whether eigen tracking should run after the completed train step."""
    eigen_tracking_every = int(getattr(cfg, "eigen_tracking_every", 100))
    if eigen_tracking_every <= 0:
        raise ValueError("eigen_tracking_every must be >= 1 when tracking is enabled.")

    if not bool(getattr(cfg, "eigen_tracking_post_soap_refresh", False)):
        return (completed_step % eigen_tracking_every) == 0

    if str(getattr(cfg, "optim", "")).lower() != "soap":
        raise ValueError(
            "eigen_tracking_post_soap_refresh=True is only supported with optim='soap'."
        )

    precondition_frequency = int(getattr(cfg, "precondition_frequency", 0))
    if precondition_frequency <= 0:
        raise ValueError(
            "eigen_tracking_post_soap_refresh=True requires precondition_frequency >= 1."
        )
    if eigen_tracking_every % precondition_frequency != 0:
        raise ValueError(
            "With eigen_tracking_post_soap_refresh=True, eigen_tracking_every must "
            "be a positive multiple of precondition_frequency."
        )

    # SOAP initializes the basis on the first optimizer step without applying an
    # actual update. The first step that uses a refreshed basis therefore occurs
    # one train step after the first QR refresh.
    first_post_refresh_step = precondition_frequency + 2
    return (
        completed_step >= first_post_refresh_step
        and ((completed_step - first_post_refresh_step) % eigen_tracking_every) == 0
    )


def construct_model(cfg):
    """Build model from config."""
    if cfg.model == "mlp":
        return MLP(num_classes=cfg.num_classes)
    elif cfg.model in {"resnet_small", "small_resnet"}:
        return SmallResNet(
            num_classes=cfg.num_classes,
            use_bn=getattr(cfg, "resnet_use_batchnorm", True),
        )
    elif cfg.model == "resnet30":
        return ResNet30(
            num_classes=cfg.num_classes,
            use_bn=getattr(cfg, "resnet_use_batchnorm", True),
        )
    elif cfg.model == "resnet18":
        return ResNet18(
            num_classes=cfg.num_classes,
            use_bn=getattr(cfg, "resnet_use_batchnorm", True),
        )
    elif cfg.model in {"vit", "vit_small", "vision_transformer"}:
        return VisionTransformer(
            num_classes=cfg.num_classes,
            patch_size=getattr(cfg, "vit_patch_size", 4),
            hidden_size=getattr(cfg, "vit_hidden_size", 128),
            mlp_dim=getattr(cfg, "vit_mlp_dim", 256),
            num_layers=getattr(cfg, "vit_layers", 4),
            num_heads=getattr(cfg, "vit_heads", 4),
            dropout_rate=getattr(cfg, "vit_dropout", 0.1),
            use_layernorm=getattr(cfg, "vit_use_layernorm", True),
            use_batchnorm=getattr(cfg, "vit_use_batchnorm", False),
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


def main(_argv):
    # Parse config
    cfg, _ = load_config(FLAGS.config)

    if cfg.model == "transformer" or str(cfg.model).startswith("pythia"):
        from train_lm import run as run_lm
        return run_lm(cfg)

    wb_run = init_wandb(cfg)
    if wb_run is not None:
        wb_run.define_metric("step")
        wb_run.define_metric("*", step_metric="step")
        wb_run.define_metric("eval_loss", summary="min")

    # Experiment dir + logging
    maybe_make_dir(cfg)
    exp_dir = get_exp_dir_path(cfg)
    writer = SummaryWriter(log_dir=exp_dir)

    # --- Optional: curvature spectrum logging ---
    log_curv = getattr(cfg, "pns_log_curvature", False)
    use_pns = cfg.optim in {
        "pns_eigenadam",
        "pns-eigenadam",
        "pns_eigenadam_adaptiv",
        "pns-eigenadam-adaptiv",
    }
    use_muon = cfg.optim in {"pns_eigenmuon", "pns-eigenmuon"}
    curvature_csv_path = None
    max_eigs = getattr(cfg, "curvature_eigenvectors", None)
    if max_eigs is None:
        max_eigs = getattr(cfg, "pns_max_eigenvectors", 16)
    max_neg_eigs = getattr(cfg, "pns_negcurv_iters", 0) or 0
    muon_max_eigs = getattr(cfg, "gradient_eigenvectors", max_eigs)


    # RNG setup
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    # Model
    model_def = construct_model(cfg)

    # Example image shape (B, H, W, C)
    image_shape = (
        cfg.batch_size,
        cfg.image_size,
        cfg.image_size,
        cfg.num_channels,
    )

    # Curvature batch for GGN / PN-EigenAdam (one deterministic batch)
    curv_train_ds, _ = get_datasets(
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        seed=cfg.seed,  # fixed seed for reproducibility
        image_size=getattr(cfg, "image_size", None),
    )
    curv_images, curv_labels = next(iter(curv_train_ds))
    curvature_batch = (curv_images, curv_labels)

    # Create train state (params + optimizer)
    state = create_train_state(
        rng=init_rng,
        model_def=model_def,
        learning_rate=cfg.lr,
        image_shape=image_shape,
        num_classes=cfg.num_classes,
        cfg=cfg,
        curvature_batch=curvature_batch,
    )

    eigen_tracking_enabled = bool(getattr(cfg, "eigen_tracking_enabled", False))
    measurement_momentum = bool(getattr(cfg, "measurement_momentum", False))
    precond_criterion_enabled = bool(getattr(cfg, "precond_criterion_enabled", False))
    precond_criterion_stratified = bool(
        getattr(cfg, "precond_criterion_stratified", True)
    )
    precond_criterion_max_dirs = int(getattr(cfg, "precond_criterion_max_dirs", 40))
    precond_criterion_damping = float(getattr(cfg, "precond_criterion_damping", 1e-6))
    eigen_tracking_state = None
    eigen_tracking_csv_path = None
    compute_probe_measurement = None

    _optim_name_lower = str(getattr(cfg, "optim", "")).lower()
    _soap_perlayer_sin2 = (
        eigen_tracking_enabled
        and _optim_name_lower in {"soap", "ni_soap", "ni-soap", "nisoap"}
        and bool(getattr(cfg, "soap_perlayer_sin2_enabled", True))
    )
    _muon_topk_energy = eigen_tracking_enabled and _optim_name_lower == "muon"
    _num_2d_layers = sum(
        1 for l in jax.tree_util.tree_leaves(state.params) if l.ndim == 2
    )

    # Effective curvature config (mirrors train_lm.py)
    effective_curvature_enabled = bool(
        getattr(cfg, "effective_curvature_tracking_enabled", False)
    )
    effective_curvature_transform_base = str(
        getattr(cfg, "effective_curvature_transform_base", "zero")
    ).lower()
    effective_curvature_direction_scale = float(
        getattr(cfg, "effective_curvature_direction_scale", 1.0)
    )
    effective_curvature_normalize_lr = bool(
        getattr(cfg, "effective_curvature_normalize_lr", True)
    )
    effective_curvature_gram_ridge = float(
        getattr(cfg, "effective_curvature_gram_ridge", 1e-8)
    )
    eigen_tracking_measurement_mode = str(
        getattr(cfg, "eigen_tracking_measurement_mode", "actual_update")
    ).lower()

    if eigen_tracking_enabled:
        eigen_tracking_topk = int(
            getattr(
                cfg,
                "eigen_tracking_topk",
                getattr(cfg, "curvature_eigenvectors", 8),
            )
        )
        if eigen_tracking_topk <= 0:
            raise ValueError("eigen_tracking_topk must be >= 1 when tracking is enabled.")
        eigen_tracking_extra_modes = int(
            getattr(cfg, "eigen_tracking_extra_modes", 0)
        )
        if eigen_tracking_extra_modes < 0:
            raise ValueError("eigen_tracking_extra_modes must be >= 0.")

        eigen_tracking_backend = getattr(
            cfg,
            "eigen_tracking_backend",
            getattr(cfg, "pns_curvature_backend", "ggn"),
        )
        eigen_tracking_iters = getattr(cfg, "eigen_tracking_lanczos_iters", None)
        if eigen_tracking_iters is None:
            eigen_tracking_iters = (
                eigen_tracking_topk + eigen_tracking_extra_modes
            )
        eigen_tracking_sort_by_abs = getattr(cfg, "eigen_tracking_sort_by_abs", None)
        if eigen_tracking_sort_by_abs is None:
            eigen_tracking_sort_by_abs = (
                str(eigen_tracking_backend).lower() in {"hessian", "fisher"}
            )
        eigen_tracking_signed_split_enabled = (
            str(eigen_tracking_backend).lower() == "hessian"
        )
        eigen_tracking_light_ortho = bool(
            getattr(cfg, "eigen_tracking_light_ortho", True)
        )
        eigen_tracking_light_ortho_every = int(
            getattr(cfg, "eigen_tracking_light_ortho_every", 4)
        )
        eigen_tracking_matvec_fn = build_curvature_matvec_fn(
            cfg,
            model_def=model_def,
            curvature_batch=curvature_batch,
            batch_stats=state.batch_stats,
            backend=eigen_tracking_backend,
        )
        _num_precond_dirs = precond_criterion_max_dirs if precond_criterion_enabled else 0
        eigen_tracking_state = init_eigentracking(
            state.params,
            k=eigen_tracking_topk,
            extra_modes=eigen_tracking_extra_modes,
            num_precond_dirs=_num_precond_dirs,
            seed=int(getattr(cfg, "seed", 0)),
        )
        eigen_tracking_csv_path = init_eigen_tracking_csv(
            cfg,
            eigen_tracking_topk,
            extra_modes=eigen_tracking_extra_modes,
            measurement_momentum=measurement_momentum,
            precond_criterion=precond_criterion_enabled,
            num_precond_dirs=_num_precond_dirs,
            soap_perlayer_sin2=_soap_perlayer_sin2,
            muon_topk_energy=_muon_topk_energy,
            num_2d_layers=_num_2d_layers,
        )

        # Capture tx and optim_name for use inside JIT (static closed-over values)
        _tx = state.tx
        _optim_name = str(getattr(cfg, "optim", "")).lower()
        _precond_criterion_damping = precond_criterion_damping

        @jax.jit
        def run_eigen_tracking(
            params, opt_state, grads, updates, step, tracking_state, precond_eigsys
        ):
            effective_transform_fn = None
            if effective_curvature_enabled:
                effective_transform_fn = _make_effective_update_transform_fn(
                    _tx,
                    params,
                    opt_state,
                    grads,
                    base_mode=effective_curvature_transform_base,
                    direction_scale=effective_curvature_direction_scale,
                    normalize_lr=effective_curvature_normalize_lr,
                    learning_rate=float(cfg.lr),
                )

            momentum_flat = None
            if measurement_momentum:
                _mom_pytree = extract_momentum_pytree(opt_state, _optim_name)
                if _mom_pytree is not None:
                    from jax.flatten_util import ravel_pytree as _rp
                    momentum_flat, _ = _rp(_mom_pytree)
                    momentum_flat = momentum_flat.astype(jnp.float32)

            return track_eigenstate(
                params=params,
                grads=grads,
                updates=updates,
                step=step,
                eigen_state=tracking_state,
                matvec_fn=eigen_tracking_matvec_fn,
                num_iter=eigen_tracking_iters,
                sort_by_abs=eigen_tracking_sort_by_abs,
                use_light_ortho=eigen_tracking_light_ortho,
                light_ortho_every=eigen_tracking_light_ortho_every,
                learning_rate=float(cfg.lr),
                signed_split_enabled=eigen_tracking_signed_split_enabled,
                effective_transform_fn=effective_transform_fn,
                effective_curvature_gram_ridge=effective_curvature_gram_ridge,
                momentum_flat=momentum_flat,
                precond_eigsys=precond_eigsys,
                precond_criterion_damping=_precond_criterion_damping,
            )

        if eigen_tracking_measurement_mode == "probe_update":
            # Weight-decay-free tx for measurements; only the update function
            # is used (with the training opt_state), no extra state allocated.
            tx_no_wd = get_optimizer(
                cfg,
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=state.batch_stats,
                override_weight_decay=0.0,
            )
            compute_probe_measurement = _make_probe_measurement_fn(
                model_def, tx_no_wd, curvature_batch, state.batch_stats
            )

    # --- Hessian probe (robustness check at ~N checkpoints) ---
    hessian_probe_enabled = bool(getattr(cfg, "hessian_probe_enabled", False))
    hessian_probe_state = None
    hessian_probe_csv_path = None
    hessian_probe_steps = set()
    if hessian_probe_enabled and eigen_tracking_enabled:
        hessian_probe_count = int(getattr(cfg, "hessian_probe_count", 5))
        hessian_probe_matvec_fn = build_curvature_matvec_fn(
            cfg,
            model_def=model_def,
            curvature_batch=curvature_batch,
            batch_stats=state.batch_stats,
            backend="hessian",
        )
        hessian_probe_state = init_eigentracking(
            state.params,
            k=eigen_tracking_topk,
            extra_modes=eigen_tracking_extra_modes,
            num_precond_dirs=0,
            seed=int(getattr(cfg, "seed", 0)) + 9999,
        )
        hessian_probe_csv_path = init_hessian_probe_csv(
            cfg,
            eigen_tracking_topk,
            extra_modes=eigen_tracking_extra_modes,
        )

        @jax.jit
        def run_hessian_probe(params, grads, updates, step, h_state):
            return track_eigenstate(
                params=params,
                grads=grads,
                updates=updates,
                step=step,
                eigen_state=h_state,
                matvec_fn=hessian_probe_matvec_fn,
                num_iter=eigen_tracking_iters,
                sort_by_abs=True,
                use_light_ortho=eigen_tracking_light_ortho,
                light_ortho_every=eigen_tracking_light_ortho_every,
                learning_rate=float(cfg.lr),
                signed_split_enabled=True,
                effective_transform_fn=None,
                effective_curvature_gram_ridge=0.0,
                momentum_flat=None,
                precond_eigsys=None,
                precond_criterion_damping=0.0,
            )

    # --- Phase-sweep measurement schedule (SOAP refresh-cycle isolation) ---
    phase_sweep_enabled = bool(getattr(cfg, "eigen_tracking_phase_sweep", False))
    phase_sweep_steps = set()
    phase_sweep_T = 0
    phase_sweep_first_fresh = 0
    if phase_sweep_enabled and eigen_tracking_enabled:
        phase_sweep_T = int(getattr(cfg, "precondition_frequency", 32))
        phase_sweep_num_cycles = int(
            getattr(cfg, "eigen_tracking_phase_sweep_num_cycles", 7)
        )
        phase_sweep_first_fresh = phase_sweep_T + 2
        offsets = [
            0,
            round(phase_sweep_T / 4),
            round(phase_sweep_T / 2),
            round(3 * phase_sweep_T / 4),
            phase_sweep_T - 1,
        ]
        _est_bpe = 100000 // cfg.batch_size
        _est_total = _est_bpe * cfg.num_epochs
        _total_cycles = max(1, (_est_total - phase_sweep_first_fresh) // phase_sweep_T)
        _cycle_interval = max(1, _total_cycles // phase_sweep_num_cycles)
        for c in range(0, _total_cycles, _cycle_interval):
            _cycle_start = phase_sweep_first_fresh + c * phase_sweep_T
            for o in offsets:
                s = _cycle_start + o
                if s <= _est_total:
                    phase_sweep_steps.add(s)
        print(
            f"Phase sweep: T={phase_sweep_T}, {len(phase_sweep_steps)} measurement steps "
            f"across {phase_sweep_num_cycles} probe cycles, "
            f"offsets={offsets}"
        )

    muon_log_files = {}
    if log_curv and use_muon:
        muon_curv_dir = os.path.join(exp_dir, "gradient_eigenvalues")
        os.makedirs(muon_curv_dir, exist_ok=True)
        muon_state = state.opt_state
        eig_leaves = [
            leaf for leaf in jtu.tree_leaves(muon_state.eigenvalues)
            if isinstance(leaf, jax.Array)
        ]
        if eig_leaves:
            muon_max_eigs = int(eig_leaves[0].shape[0])
        header = ["epoch", "global_step"] + [f"eig_{i}" for i in range(muon_max_eigs)]

        def register_layer(path, leaf):
            if hasattr(leaf, "ndim") and leaf.ndim == 2:
                layer_name = "/".join(str(k) for k in path)
                file_name = f"{_sanitize_name(layer_name)}.csv"
                file_path = os.path.join(muon_curv_dir, file_name)
                muon_log_files[layer_name] = file_path
                with open(file_path, "w") as f:
                    f.write(",".join(header) + "\n")
            return None

        jtu.tree_map_with_path(register_layer, state.params)

    if log_curv and use_pns:
        curvature_csv_path = os.path.join(exp_dir, "curvature.csv")
        pns_state = state.opt_state
        if hasattr(pns_state, "eigenvalues"):
            max_eigs = int(jnp.asarray(pns_state.eigenvalues).shape[0])
        header = ["epoch", "global_step"] + [f"eig_{i}" for i in range(max_eigs)]
        if max_neg_eigs > 0:
            header += (
                ["rotation_diff_pos"]
                + [f"eig_neg_{i}" for i in range(max_neg_eigs)]
                + ["rotation_diff_neg"]
            )
        else:
            header += ["rotation_diff_pos"]
        with open(curvature_csv_path, "w") as f:
            f.write(",".join(header) + "\n")

    # --------- OPTIONAL: profile PN-S curvature step once ---------
    # Only makes sense if you're using a curvature-based optimizer
    # (pns_eigenadam, pns_eigenmuon, curvature_muon, hf, etc.)
    #if cfg.optim in {"pns_eigenadam", "pns-eigenadam",
    #                 "pns_eigenmuon", "pns-eigenmuon",
    #                 "curvature_muon", "curvature-muon"}:
    #    # Build the same GGN matvec that the optimizer factory uses
    #    ggn_mv = make_ggn_matvec_fn(
    #        model_def=model_def,
    #        curvature_batch=curvature_batch,
    #        batch_stats=state.batch_stats,
    #    )
#
    #    # Use config knobs if they exist, else fall back to defaults
    #    max_eigs = getattr(cfg, "pns_max_eigenvectors", 16)
    #    lanczos_iters = getattr(cfg, "pns_lanczos_iters", None)
#
    #    # Use the main rng or split a fresh one
    #    rng, prof_rng = jax.random.split(rng)
#
    #    profile_pns_eigenadam_curvature(
    #        params=state.params,
    #        ggn_matvec_fn=ggn_mv,
    #        max_eigenvectors=max_eigs,
    #        lanczos_iters=lanczos_iters,
    #        rng=prof_rng,
    #        warmup=True,
    #    )
    ## --------------------------------------------------------------

    train_step = make_train_step(return_updates=False)
    tracking_train_step = (
        make_train_step(return_updates=True) if eigen_tracking_enabled else None
    )
    eval_step = make_eval_step()

    # For curves: wall-clock vs loss and iteration vs loss
    wall_times = []
    iters = []
    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []

    # Pre-compute which tracking steps get a Hessian probe
    if hessian_probe_enabled and eigen_tracking_enabled:
        _eigen_every = int(getattr(cfg, "eigen_tracking_every", 100))
        _est_batches_per_epoch = 100000 // cfg.batch_size  # TinyImageNet: 100k train
        _est_total_steps = _est_batches_per_epoch * cfg.num_epochs
        _est_total_trackings = max(1, _est_total_steps // _eigen_every)
        _probe_interval = max(1, _est_total_trackings // hessian_probe_count)
        _tracking_counter = [0]  # mutable counter for tracking steps seen

    training_start_time = time.time()
    train_global_step = 0
    # Running best, logged every epoch so the run summary ends up holding a
    # plain float. W&B sweeps read the objective off the summary, and
    # define_metric(..., summary="min") stores eval_loss as {"min": ...}, which
    # does not resolve to a number -- see config/*_sweep_*.yaml.
    best_eval_loss = float("inf")

    # --------------------
    # Training loop
    # --------------------
    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        # Fresh dataset iterators each epoch
        train_ds, test_ds = get_datasets(
            dataset=cfg.dataset,
            batch_size=cfg.batch_size,
            seed=cfg.seed + epoch,  # optional: vary seed per epoch
            image_size=getattr(cfg, "image_size", None),
        )

        # ---- Train ----
        train_metrics = []
        for batch in train_ds:
            rng, batch_rng = jax.random.split(rng)
            _completed = train_global_step + 1
            if phase_sweep_enabled:
                should_track = eigen_tracking_enabled and (_completed in phase_sweep_steps)
            else:
                should_track = (
                    eigen_tracking_enabled
                    and _should_run_eigen_tracking_for_step(cfg, _completed)
                )
            if should_track:
                params_before = state.params
                opt_state_before = state.opt_state
                state, metrics, grads, updates = tracking_train_step(
                    state, batch, batch_rng
                )

                measurement_metrics = {}
                if compute_probe_measurement is not None:
                    measurement_metrics = jax.device_get(
                        compute_probe_measurement(params_before, opt_state_before)
                    )

                if phase_sweep_enabled:
                    measurement_metrics["refresh_T"] = phase_sweep_T
                    measurement_metrics["refresh_offset"] = (
                        (_completed - phase_sweep_first_fresh) % phase_sweep_T
                    )

                train_ds_meas, eval_ds_meas = get_datasets(
                    dataset=cfg.dataset,
                    batch_size=cfg.batch_size,
                    seed=cfg.seed,
                    image_size=getattr(cfg, "image_size", None),
                )
                _eval_metrics_meas = [eval_step(state, batch) for batch in eval_ds_meas]
                measurement_metrics["eval_loss"] = float(
                    jnp.mean(jnp.array([m["loss"] for m in _eval_metrics_meas]))
                )
                measurement_metrics["eval_accuracy"] = float(
                    jnp.mean(jnp.array([m["accuracy"] for m in _eval_metrics_meas]))
                )

                # Train-side loss on the same probe grid, measured with the same
                # estimator (eval_step -> train=False, so no dropout) over as
                # many train batches as the eval split has. The split is not
                # augmented and get_datasets is seeded, so this is the same
                # subset at every checkpoint and eval_loss - train_loss_probe is
                # a like-for-like generalisation gap.
                _n_meas_batches = len(_eval_metrics_meas)
                _train_metrics_meas = []
                for _meas_batch in train_ds_meas:
                    if len(_train_metrics_meas) >= _n_meas_batches:
                        break
                    _train_metrics_meas.append(eval_step(state, _meas_batch))
                if _train_metrics_meas:
                    measurement_metrics["train_loss_probe"] = float(
                        jnp.mean(jnp.array([m["loss"] for m in _train_metrics_meas]))
                    )
                    measurement_metrics["train_accuracy_probe"] = float(
                        jnp.mean(jnp.array([m["accuracy"] for m in _train_metrics_meas]))
                    )

                _precond_eigsys = None
                _precond_meta = None
                if precond_criterion_enabled:
                    _precond_dirs = extract_precond_directions(
                        opt_state_before,
                        _optim_name,
                        params_before,
                        precond_criterion_max_dirs,
                        stratified=precond_criterion_stratified,
                    )
                    if _precond_dirs is not None:
                        _, _, _pc_dim = _leaf_offsets_of(params_before)
                        _precond_eigsys = densify_precond_directions(
                            _precond_dirs, _pc_dim
                        )
                        # a_d/phat_d come from the in-JIT path; lambda_hat and the
                        # stratum label are only known here, at extraction time.
                        _precond_meta = {
                            "lambda_hat_per_dir": _precond_dirs.lambda_hat,
                            "stratum_per_dir": _precond_dirs.stratum,
                        }
                eigen_tracking_state = run_eigen_tracking(
                    params_before,
                    opt_state_before,
                    grads,
                    updates,
                    state.step,
                    eigen_tracking_state,
                    _precond_eigsys,
                )

                if _soap_perlayer_sin2:
                    _is_ni = _optim_name in {"ni_soap", "ni-soap", "nisoap"}
                    _sin2_mean, _sin2_layers = compute_soap_perlayer_sin2(
                        opt_state_before, _is_ni, params_before,
                        eigen_tracking_state.eigenvectors,
                    )
                    measurement_metrics["soap_sin2_layermean"] = _sin2_mean
                    for _i, _s in enumerate(_sin2_layers):
                        measurement_metrics[f"soap_sin2_layer_{_i}"] = _s
                elif _muon_topk_energy:
                    _e_mean, _e_base, _e_layers, _b_layers = compute_muon_topk_energy(
                        updates, params_before,
                        eigen_tracking_state.eigenvectors,
                    )
                    measurement_metrics["muon_topk_energy_mean"] = _e_mean
                    measurement_metrics["muon_topk_energy_baseline_mean"] = _e_base
                    for _i, (_e, _b) in enumerate(
                        zip(_e_layers, _b_layers)
                    ):
                        measurement_metrics[f"muon_topk_energy_layer_{_i}"] = _e
                        measurement_metrics[f"muon_topk_energy_baseline_layer_{_i}"] = _b

                append_eigen_tracking_row(
                    eigen_tracking_csv_path,
                    eigen_tracking_state,
                    measurement_metrics=measurement_metrics,
                    measurement_momentum=measurement_momentum,
                    precond_criterion=precond_criterion_enabled,
                    num_precond_dirs=_num_precond_dirs,
                    soap_perlayer_sin2=_soap_perlayer_sin2,
                    muon_topk_energy=_muon_topk_energy,
                    num_2d_layers=_num_2d_layers,
                    precond_metrics=_precond_meta,
                )

                if hessian_probe_enabled:
                    _tracking_counter[0] += 1
                    if (_tracking_counter[0] - 1) % _probe_interval == 0:
                        hessian_probe_state = run_hessian_probe(
                            params_before, grads, updates,
                            state.step, hessian_probe_state,
                        )
                        append_hessian_probe_row(
                            hessian_probe_csv_path, hessian_probe_state,
                        )
            else:
                state, metrics = train_step(state, batch, batch_rng)
            train_global_step += 1
            train_metrics.append(metrics)

        # ---- Eval ----
        eval_metrics = []
        for batch in test_ds:
            metrics = eval_step(state, batch)
            eval_metrics.append(metrics)

        def stack_metrics(metrics_list):
            return {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                for k in metrics_list[0]
            }

        train_summary = stack_metrics(train_metrics)
        eval_summary = stack_metrics(eval_metrics)

        epoch_time = time.time() - start_time

        # Accumulate data for curves
        # inside epoch loop, after computing train_summary / eval_summary:
        elapsed_from_start = time.time() - training_start_time
        wall_times.append(float(elapsed_from_start))
        iters.append(int(epoch))
        train_losses.append(float(train_summary["loss"]))
        eval_losses.append(float(eval_summary["loss"]))
        train_accs.append(float(train_summary["accuracy"]))
        eval_accs.append(float(eval_summary["accuracy"]))

        metrics = {
                # define_metric("*", step_metric="step") above makes every chart
                # plot against "step", so it has to be logged -- without it W&B
                # fills 0 for every epoch and all curves collapse onto x=0.
                "step": train_global_step,
                "epoch": epoch,
                "train_loss": float(train_summary["loss"]),
                "train_accuracy": float(train_summary["accuracy"]),
                "eval_loss": float(eval_summary["loss"]),
                "eval_accuracy": float(eval_summary["accuracy"]),
                "epoch_time": epoch_time,
            }
        best_eval_loss = min(best_eval_loss, float(eval_summary["loss"]))
        metrics["best_eval_loss"] = best_eval_loss
        # Console + optional wandb logging
        log_scalar_dict(
            cfg,
            metrics,
        )

        if log_curv and use_pns and curvature_csv_path is not None:
            pns_state = state.opt_state 

            # Positive spectrum
            ev_pos = jnp.asarray(pns_state.eigenvalues)
            rot_pos = float(pns_state.rotation_diff)
            step_opt = int(pns_state.step)
            eig_pos_list = [float(x) for x in ev_pos[:max_eigs]]

            # Negative spectrum (if present)
            if max_neg_eigs > 0:
                ev_neg = jnp.asarray(pns_state.neg_eigenvalues)
                rot_neg = float(pns_state.neg_rotation_diff)
                eig_neg_list = [float(x) for x in ev_neg[:max_neg_eigs]]
                row = (
                    [epoch, step_opt]
                    + eig_pos_list
                    + [rot_pos]
                    + eig_neg_list
                    + [rot_neg]
                )
            else:
                row = [epoch, step_opt] + eig_pos_list + [rot_pos]

            with open(curvature_csv_path, "a") as f:
                f.write(",".join(str(x) for x in row) + "\n")

        if log_curv and use_muon and muon_log_files:
            muon_state = state.opt_state
            step_opt = int(muon_state.step)

            def write_muon_row(path, eigs):
                if eigs is None:
                    return None
                layer_name = "/".join(str(k) for k in path)
                file_path = muon_log_files.get(layer_name)
                if file_path is None:
                    return None
                eig_vals = [float(x) for x in jnp.asarray(eigs)[:muon_max_eigs]]
                row = [epoch, step_opt] + eig_vals
                with open(file_path, "a") as f:
                    f.write(",".join(str(x) for x in row) + "\n")
                return None

            jtu.tree_map_with_path(write_muon_row, muon_state.eigenvalues)


        # TensorBoard logging
        writer.scalar("train/loss", float(train_summary["loss"]), step=epoch)
        writer.scalar("train/accuracy", float(train_summary["accuracy"]), step=epoch)
        writer.scalar("eval/loss", float(eval_summary["loss"]), step=epoch)
        writer.scalar("eval/accuracy", float(eval_summary["accuracy"]), step=epoch)
        writer.scalar("epoch_time", float(epoch_time), step=epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_summary['loss']:.4f}, "
            f"train acc {train_summary['accuracy']:.4f} | "
            f"eval loss {eval_summary['loss']:.4f}, "
            f"eval acc {eval_summary['accuracy']:.4f} | "
            f"time {epoch_time:.2f}s"
        )

    # Save CSVs + plots for wall-clock vs loss and iteration vs loss
    save_loss_curves(
        cfg=cfg,
        optimizer_name=cfg.optim,
        wall_times=wall_times,
        iterations=iters,
        train_losses=train_losses,
        eval_losses=eval_losses,
        train_accuracies=train_accs,
        eval_accuracies=eval_accs,
    )

    writer.flush()
    writer.close()
    
    

if __name__ == "__main__":
    app.run(main)
