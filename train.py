import functools

from image_converter_utils import save_xyb, save_rgb_image
from losses import loss_fn
import optax
import jax
import jax.numpy as jnp
from optimizer_values import RGBOptimizerValues, XYBOptimizerValues, XYBDCTOptimizerValues
from codex.loss import load_vgg16_model

# Cache jit compilation
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
jax.config.update("jax_debug_nans", True)

# Training step
@functools.partial(jax.jit, static_argnames=['use_l2'])
def train_step(
        optimizer_values,
        target,
        target_features,
        log2_sigma,
        opt_state,
        lambd,
        gamma,
        xyb_multiplier_dct,
        xyb_multiplier,
        l2_xyb_multiplier,
        use_l2,
):
    (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        optimizer_values,
        target,
        target_features,
        log2_sigma,
        lambd,
        gamma,
        xyb_multiplier_dct,
        xyb_multiplier,
        l2_xyb_multiplier,
        use_l2,
    )
    updates, opt_state = optimizer.update(grad, opt_state, params=optimizer_values)
    optimizer_values = optax.apply_updates(optimizer_values, updates)
    return optimizer_values, opt_state, loss


def create_image_split(
        target,
        target_features,
        lambd,
        gamma,
        log2_sigma_value,
        l2_loops,
        ws_loops,
        xyb_multiplier_dct,
        xyb_multiplier,
        l2_rgb_multiplier,
        layers,
        index,
        base
):
    opt_params = {
        (True,): {"lr": 5e-3, "wd": 0.0, "steps": l2_loops},
        (False,): {"lr": 5e-3, "wd": 0.0, "steps": ws_loops},
    }
    # Initialize a candidate image
    H, W, C = target.shape  # target is (H, W, C)

    if base == "rgb":
        candidate = RGBOptimizerValues(target.shape, layers)
    elif base == "xyb":
        candidate = XYBOptimizerValues(target.shape, layers)
    elif base == "dct":
        candidate = XYBDCTOptimizerValues(target.shape, layers)
    else:
        raise Exception(f"Unknown base type: {base}")
    log2_sigma = jnp.full((H, W), log2_sigma_value, dtype=jnp.float32)

    loss, (ws_loss, compression_losses) = loss_fn(
        candidate,
        target,
        target_features,
        log2_sigma,
        lambd,
        gamma,
        xyb_multiplier_dct,
        xyb_multiplier,
        l2_rgb_multiplier,
        True,
    )
    print(
        f"At init: loss {loss}, l2 loss {ws_loss}, closs {sum(compression_losses)} ({jnp.array(compression_losses).tolist()})"
    )

    loss, (ws_loss, compression_losses) = loss_fn(
        candidate,
        target,
        target_features,
        log2_sigma,
        lambd,
        gamma,
        xyb_multiplier_dct,
        xyb_multiplier,
        l2_rgb_multiplier,
        False,
    )
    print(
        f"At init: loss {loss}, wloss {ws_loss}, closs {sum(compression_losses)} ({jnp.array(compression_losses).tolist()})"
    )

    last_kind = None
    opt_state = None
    for step in range(l2_loops + ws_loops):
        use_l2 = step < l2_loops
        kind = (use_l2,)

        if kind != last_kind:
            global optimizer
            opt_params_current = opt_params[kind]
            schedule = optax.schedules.cosine_decay_schedule(
                init_value=opt_params_current["lr"], decay_steps=opt_params_current["steps"]
            )
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(schedule, weight_decay=opt_params_current["wd"]),
            )

            opt_state = optimizer.init(candidate)
            last_kind = kind

        candidate, opt_state, loss = train_step(
            candidate,
            target,
            target_features,
            log2_sigma,
            opt_state,
            lambd,
            gamma,
            xyb_multiplier_dct,
            xyb_multiplier,
            l2_rgb_multiplier,
            use_l2,
        )

        if step % 200 == 0:
            loss, (ws_loss, compression_losses) = loss_fn(
                candidate,
                target,
                target_features,
                log2_sigma,
                lambd,
                gamma,
                xyb_multiplier_dct,
                xyb_multiplier,
                l2_rgb_multiplier,
                use_l2,
            )
            print(
                f"Step {step + 1}, loss {loss}, {'l2 loss' if step < l2_loops else 'wloss'} {ws_loss}, closs {sum(compression_losses)} ({jnp.array(compression_losses).tolist()})"
            )

    loss, (ws_loss, compression_losses) = loss_fn(
        candidate,
        target,
        target_features,
        log2_sigma,
        lambd,
        gamma,
        xyb_multiplier_dct,
        xyb_multiplier,
        l2_rgb_multiplier,
        False
    )

    for i, up_candidate in enumerate(candidate.convert_to_xyb()):
        save_xyb(
            up_candidate,
            f"out/reconstructed_image_xyb_{index}_{i}.txt",
        )
    for i, up_candidate in enumerate(candidate.convert_to_rgb()):
        save_rgb_image(up_candidate, f"out/reconstructed_image_{index}_{i}.png")

    save_rgb_image(candidate.combine_to_rgb(), f"out/reconstructed_image_{index}_combined.png")

    print(
        f"creating out/reconstructed_image_{index}_combined.png, wasserstein loss {ws_loss}, compression loss {sum(compression_losses)} ({jnp.array(compression_losses).tolist()})"
    )
    return ws_loss, compression_losses
