import dataclasses as dc


@dc.dataclass
class DGArguments:
    model: str = dc.field(
        default="KB/bert-base-swedish-cased",
        metadata={"help": "BERT model to load"}
    )

    train_data: str = dc.field(
        default="data/training.json",
        metadata={"help": "A CSV list of training data files"}
    )

    dev_data: str = dc.field(
        default=None,
        metadata={"help": "A CSV list of validation data files"}
    )

    freeze_encoder: bool = dc.field(
        default=False,
        metadata={"help": "Freezes BERT encoder"}
    )

    formulation: str = dc.field(
        default="areg",
        metadata={"help": "Type of problem definition: autoregressive (areg) or u-PMLM (upmlm)"}
    )

    upmlm_samples: int = dc.field(
        default=20,
        metadata={"help": "Number of samples for u-PMLM (active only if formulation is upmlm)"}
    )

    upmlm_seed: int = dc.field(
        default=42,
        metadata={"help": "Random seed for u-PMLM masking scheme"}
    )

