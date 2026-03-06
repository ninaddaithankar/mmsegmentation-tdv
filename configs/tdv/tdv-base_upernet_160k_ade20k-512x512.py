_base_ = [
    '../_base_/models/upernet_tdv.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # Set checkpoint_path to your TDV .ckpt file, e.g.:
        #   checkpoint_path='/path/to/your/tdv_checkpoint.ckpt'
        # Leave as None to use pretrained DINOv2 weights from torch hub.
        checkpoint_path=None,
        frozen=True),
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# -----------------------------------------------------------------------
# Optimiser  –  only the neck + heads are trainable (backbone is frozen)
# -----------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False),
]

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
