from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.structured import ProjectConfig


def _load_dependencies():
    from .model import ConditionalPointCloudDiffusionModel
    from .model_coloring import PointCloudColoringModel
    from .model_utils import set_requires_grad
    from .model_diff_data import ConditionalPCDiffusionSeparateSegm

    return (
        ConditionalPointCloudDiffusionModel,
        PointCloudColoringModel,
        set_requires_grad,
        ConditionalPCDiffusionSeparateSegm,
    )


def get_model(cfg: "ProjectConfig"):
    (
        ConditionalPointCloudDiffusionModel,
        _PointCloudColoringModel,
        set_requires_grad,
        ConditionalPCDiffusionSeparateSegm,
    ) = _load_dependencies()
    if cfg.model.model_name == 'pc2-diff':
        model = ConditionalPointCloudDiffusionModel(**cfg.model)
    elif cfg.model.model_name == 'pc2-diff-ho-sepsegm':
        model = ConditionalPCDiffusionSeparateSegm(**cfg.model)
        print("Using a separate model to predict segmentation label")
    else:
        raise NotImplementedError
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model


def get_coloring_model(cfg: "ProjectConfig"):
    (
        _ConditionalPointCloudDiffusionModel,
        PointCloudColoringModel,
        set_requires_grad,
        _ConditionalPCDiffusionSeparateSegm,
    ) = _load_dependencies()
    model = PointCloudColoringModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model
