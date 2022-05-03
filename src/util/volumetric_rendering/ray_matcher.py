import typing as t

import torch
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)


class Fragments(t.NamedTuple):
    features: torch.Tensor
    opacities: torch.Tensor
    weights: torch.Tensor
    absorption: torch.Tensor


class EmissionAbsorptionRaymarcherWithFragments(torch.nn.Module):
    """
    Raymarch using the Emission-Absorption (EA) algorithm.

    Standard implementation of ray marcher taken from Pytorch3D library, however
    this implementation also returns intermediate computations.
    """

    def __init__(self, surface_thickness: int = 1) -> None:
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        super().__init__()
        self.surface_thickness = surface_thickness

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> Fragments:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.

        Returns:
            features_opacities: A Fragments instance with following components
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
                    3) weights: A tensor of per-ray-per-point weights values of
                        shape `(..., n_rays, n_points_per_ray)`.
                    4) absorption: A tensor with absorption values
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)

        return Fragments(
            features=features,
            opacities=opacities,
            weights=weights,
            absorption=absorption,
        )
