from model import unet_model
from utils import *
from torch.optim import Adam
from dataset import CellDataset
import tqdm


def train(model):
    for epoch in range(num_epochs):
        print(f"====>> {epoch} <<====")
        loop = tqdm(enumerate(train_batch), total=len(train_batch))
        for batch_idx, (data, targets) in loop:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        check_accuracy(test_batch, model, epoch + 1)
        save_checkpoint(model)


data = CellDataset(DATA_LIST, transform)
train_dataset, test_dataset = torch.utils.data.random_split(
    data, [TRAIN_SIZE, TEST_SIZE]
)
train_batch = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
)
test_batch = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False
)

model = unet_model(in_channels=IN_CHANNELS).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
