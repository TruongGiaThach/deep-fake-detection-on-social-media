import os
import warnings
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.transforms import ToPILImage, ToTensor
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from PIL import ImageChops, Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import LABEL_FILE, FRAME_SAVE_PATH, MODEL_SAVE_PATH, MODEL_CHECKPOINT, LOGS_PATH
from models.data_provider import DeepfakeDataset
from data_utils.data import load_face
from models.efficient_net_lstm import EfficientNetLSTM

# Config
BATCH_SIZE = 10
LEARNING_RATE = 1e-4

def main():
    # Config
    train_from_scratch = False
    initial_lr = LEARNING_RATE
    min_lr = initial_lr * 1e-5
    log_interval = 74
    validation_interval = 749
    max_num_iterations = 20000
    val_loss = min_val_loss = patience = 10
    epoch = iteration = 0
    face_size = 128
    face_policy = 'scale'
    model_state = None
    opt_state = None
    logs_folder = "logs"
    tag = "efficient_netb0"
    enable_attention = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = EfficientNetLSTM().to(device)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Flip images randomly
        transforms.RandomRotation(10),  # Rotate slightly
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color adjustments
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset from CSV
    dataset = DeepfakeDataset(LABEL_FILE, FRAME_SAVE_PATH, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    if len(val_dataset) == 0:
        print('No validation samples. Halt.')
        return
    
    # Initialize Tensorboard
    logdir = os.path.join(logs_folder, tag)
    if iteration == 0:
        # If training from scratch or initialization remove history if exists
        shutil.rmtree(logdir, ignore_errors=True)

    # TensorboardX instance
    tb = SummaryWriter(logdir=logdir)
    if iteration == 0:
        dummy = torch.randn((BATCH_SIZE, 1, 3, face_size, face_size), device=device)
        dummy = dummy.to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tb.add_graph(model, [dummy, ], verbose=False)
        
    # Compute class weights
    labels_df = pd.read_csv(LABEL_FILE)
    labels_np = labels_df["label"].values
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_np), y=labels_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    torch.set_num_threads(os.cpu_count())  # Use all available CPU threads

    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None # Mixed precision training (if using GPU)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=min_lr,
    )

    # Model checkpoint paths
    bestval_path = os.path.join(MODEL_CHECKPOINT, 'bestval.pth')
    last_path = os.path.join(MODEL_CHECKPOINT, 'last.pth')
    periodic_path = os.path.join(MODEL_CHECKPOINT, 'it{:06d}.pth')

    if not train_from_scratch and os.path.exists(last_path):
        print('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        model_state = state['model']
        opt_state = state['opt']
        iteration = state['iteration'] + 1
        epoch = state['epoch']
        print(f"Resuming from epoch {epoch}")

    if not train_from_scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']
    if model_state is not None:
        incomp_keys = model.load_state_dict(model_state, strict=False)
        print(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = initial_lr
        optimizer.load_state_dict(opt_state)

    # Training Loop
    stop = False
    while not stop:
        model.train()
        total_loss, correct, total, = 0, 0, 0
        train_loss = train_num = correct = 0
        all_labels = []
        all_preds = []
        
        loop = tqdm(train_loader, leave=True)  # Add tqdm progress bar
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            train_batch_loss, train_batch_pred = batch_forward(model, device, criterion, images, labels, scaler=scaler)

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')
            
            train_batch_num = len(images)
            train_num += train_batch_num

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(train_batch_pred.cpu().numpy())
            
            train_loss += train_batch_loss.item() * train_batch_num

            # Update tqdm description
            total += labels.size(0)
            total_loss += train_batch_loss.item()
            correct += (train_batch_pred == labels).sum().item()
            train_acc = correct / total
            loop.set_description(f"Epoch [{epoch+1}]")
            loop.set_postfix(loss=total_loss/total, acc=train_acc)

            # Optimization
            if scaler is not None:
                scaler.scale(train_batch_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                train_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
            # Logging
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                tb.add_scalar('epoch', epoch, iteration)
                tb.add_scalar('train/total', total, iteration)
                tb.add_scalar('train/total_loss', total_loss, iteration)
                tb.add_scalar('train/total_correct', correct, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                
                # Checkpoint
                save_model(model, optimizer, train_loss, val_loss, iteration, BATCH_SIZE, epoch, last_path)
                train_loss = train_num = 0

            # Validation
            if iteration > 0 and (iteration % validation_interval == 0):
                # Model checkpoint
                save_model(model, optimizer, train_loss, val_loss, iteration, BATCH_SIZE, epoch, periodic_path.format(iteration))

                train_labels = all_labels
                train_pred = all_preds
                all_labels = []
                all_preds = []

                train_roc_auc = roc_auc_score(train_labels, train_pred)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.add_pr_curve('train/pr', train_labels, train_pred, iteration)

                # Validation
                val_loss = validation_routine(model, device, val_loader, criterion, tb, iteration, 'val', scaler)
                tb.flush()

                # LR Scheduler
                lr_scheduler.step(val_loss)
                
                # Save best Model
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model(model, optimizer, train_loss, val_loss, iteration, BATCH_SIZE, epoch, bestval_path)

                # Attention
                if enable_attention and hasattr(model, 'get_attention'):
                    model.eval()
                    # Load label file to find frame paths
                    labels_df = pd.read_csv(LABEL_FILE)

                    # Get one real and one fake sample
                    real_idx = labels_df[labels_df['label'] == 0].index[0]
                    fake_idx = labels_df[labels_df['label'] == 1].index[0]

                    for sample_idx, tag in [(real_idx, 'train/att/real'), (fake_idx, 'train/att/fake')]:
                        record = labels_df.loc[sample_idx]
                        tb_attention(tb, tag, iteration, model, device, face_size, face_policy, transform, FRAME_SAVE_PATH, record)

                if optimizer.param_groups[0]['lr'] == min_lr:
                    print('Reached minimum learning rate. Stopping.')
                    stop = True
                    break

            iteration += 1

            if iteration > max_num_iterations:
                print('Maximum number of iterations reached')
                stop = True
                break

            # End of iteration

        epoch += 1 

def tb_attention(tb: SummaryWriter,
                 tag: str,
                 iteration: int,
                 model: nn.Module,
                 device: torch.device,
                 patch_size_load: int,
                 face_crop_scale: str,
                 val_transformer: A.BasicTransform,
                 root: str,
                 record: pd.Series,
                 ):
    # Crop face
    sample_t = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                         transformer=val_transformer)
    sample_t_clean = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                               transformer=ToTensorV2())
    if torch.cuda.is_available():
        sample_t = sample_t.cuda(device)
    # Transform
    # Feed to net
    with torch.no_grad():
        att: torch.Tensor = model.get_attention(sample_t.unsqueeze(0))[0].cpu()
    att_img: Image.Image = ToPILImage()(att)
    sample_img = ToPILImage()(sample_t_clean)
    att_img = att_img.resize(sample_img.size, resample=Image.NEAREST).convert('RGB')
    sample_att_img = ImageChops.multiply(sample_img, att_img)
    sample_att = ToTensor()(sample_att_img)
    tb.add_image(tag=tag, img_tensor=sample_att, global_step=iteration)

def batch_forward(model: EfficientNetLSTM, device: torch.device, criterion, data, labels, scaler=None):
    data = data.to(device)
    labels = labels.to(device)

    with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
        out = model(data)
        loss = criterion(out, labels)
    _, pred = torch.max(out, 1)
    return loss, pred  

def validation_routine(model, device, val_loader, criterion, tb, iteration, tag:str, scaler, loader_len_norm: int = None):
    model.eval()
    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader) // loader_len_norm):
        batch_data, batch_labels = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(model, device, criterion, batch_data, batch_labels, scaler=scaler)

        pred_list.append(val_batch_pred.flatten())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num
        val_correct += (val_batch_pred == batch_labels).sum().item()

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/total'.format(tag), val_num, iteration)
    tb.add_scalar('{}/total_loss'.format(tag), val_loss, iteration)
    tb.add_scalar('{}/total_correct'.format(tag), val_correct, iteration)

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        val_labels = np.concatenate(labels_list)
        val_pred = np.concatenate(pred_list)
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
        tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss


def save_model(model: EfficientNetLSTM, optimizer: optim.Optimizer,
               train_loss: float, val_loss: float,
               iteration: int, batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(model=model.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


if __name__ == '__main__':
    main()