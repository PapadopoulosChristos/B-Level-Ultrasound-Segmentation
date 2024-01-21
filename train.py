import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics, losses
from torch.utils.data import DataLoader
from data import UltrasoundDataset

device = torch.device("cuda")

# Create Datasets and DataLoaders.
train_data = UltrasoundDataset(
    data_dir="/home/c/user/Desktop/misc", split="train"
)
val_data = UltrasoundDataset(
    data_dir="/home/c/user/Desktop/misc", split="val"
)

train_loader = DataLoader(
    train_data, batch_size=8, shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    val_data, batch_size=1, shuffle=False, num_workers=0
)


# Create model, criterion, metrics, and optimizer objects.
model = smp.UnetPlusPlus(
    encoder_name="se_resnext50_32x4d",
    encoder_depth=5,
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    activation="simgoid",
)

criterion = losses.DiceLoss()
metrics = [metrics.IoU(threshold=0.5)]
optim = torch.optim.Adam(model.parameters(), lr=0.0001)


# Create objects to run training and validation steps.
training_run = smp.utils.train.TrainEpoch(
    model, 
    loss=criterion, 
    metrics=metrics, 
    optimizer=optim,
    device=device,
    verbose=True,
)

validation_run = smp.utils.train.ValidEpoch(
    model, 
    loss=criterion, 
    metrics=metrics, 
    device=device,
    verbose=True,
)

# Training.
print("Commencting training...")
max_score = 0
for i in range(0, 50):
    
    print('\nEpoch: {}'.format(i))
    train_res = training_run.run(train_loader)
    val_res = validation_run.run(valid_loader)
    
    # Keep best model.
    if max_score < val_res['iou_score']:
        max_score = val_res['iou_score']
        torch.save(model, './best.pth')
        print('Model saved!')
    
    # Reduce learning rate in intervals.
    if i == 15:
        optim.param_groups[0]['lr'] = 1e-5
        print('Decrease LR to 1e-5!')
    if i == 30:
        optim.param_groups[0]['lr'] = 1e-6
        print('Decrease LR to 1e-6!')
        
print("Training Complete!")