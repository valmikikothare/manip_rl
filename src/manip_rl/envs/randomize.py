"""Domain randomization for manipulation envs.

Follows the Playground randomizer signature:
    randomizer(model: mjx.Model, rng: jax.Array) -> (batched_model, in_axes)
where `rng` has shape (num_worlds, 2) and the returned model is vmapped over
the randomized fields. Wire into training via
`wrap_for_brax_training(..., randomization_fn=functools.partial(domain_randomize, env=env))`.
"""

import jax
import jax.numpy as jp
from mujoco import mjx


def domain_randomize(model: mjx.Model, rng: jax.Array, env=None):
    """Randomize object physics + actuator gains per world.

    Ranges are multiplicative and deliberately moderate; tune per task.
    Object randomization targets `env.obj_body_id`; when called through the
    Playground registry (no env arg), a default PandaPickPlace resolves the id.
    """
    if env is None:
        from manip_rl.envs.registry import PandaPickPlace
        env = PandaPickPlace()

    @jax.vmap
    def rand(rng):
        rng_friction, rng_mass, rng_gain = jax.random.split(rng, 3)

        # Sliding friction for all geoms: x0.6 .. x1.4
        friction_scale = jax.random.uniform(rng_friction, minval=0.6, maxval=1.4)
        geom_friction = model.geom_friction.at[:, 0].mul(friction_scale)

        # Object mass (and inertia with it): x0.5 .. x2.0 (log-uniform)
        body_mass = model.body_mass
        body_inertia = model.body_inertia
        if env is not None:
            mass_scale = jp.exp(jax.random.uniform(
                rng_mass, minval=jp.log(0.5), maxval=jp.log(2.0)))
            body_mass = body_mass.at[env.obj_body_id].mul(mass_scale)
            body_inertia = body_inertia.at[env.obj_body_id].mul(mass_scale)

        # Actuator gain: x0.9 .. x1.1
        gain_scale = jax.random.uniform(rng_gain, minval=0.9, maxval=1.1)
        actuator_gainprm = model.actuator_gainprm.at[:, 0].mul(gain_scale)

        return geom_friction, body_mass, body_inertia, actuator_gainprm

    geom_friction, body_mass, body_inertia, actuator_gainprm = rand(rng)

    fields = {
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_inertia": body_inertia,
        "actuator_gainprm": actuator_gainprm,
    }
    in_axes = jax.tree.map(lambda x: None, model)
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    model = model.tree_replace(fields)
    return model, in_axes
