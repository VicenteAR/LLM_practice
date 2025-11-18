import torch
from gpt import (
    TrainingText,
    SetSplit,
    BiagramLanguageModel,
    SingleHeadBiagramLanguageModel,
    MultiHeadBiagramLanguageModel,
    GetBatch,
    EstimateLoss,
)

# Get training text. Generate decoder/encoder and vocab paramenters
file = TrainingText(
    input_file=r"/Users/vicentearjona/Documents/LLM_practice/Neural_Networks/input.txt"
)
# Parameters
vocab_size = file.vocab_size
context_size = 16
batch_size = 16
max_iters = 10000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
head_size = 64
head_dim = 8
emb_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
# Set seed
torch.manual_seed(1337)
# Transform into torch tensor
data = torch.tensor(file.encoder(file.text), dtype=torch.long)
# Get train / val / test splits
splitter = SetSplit(train_size=0.75, test_size=0.15)
train, val, test = splitter.split(data=data)
# Define the object to split the data into batches
get_batch = GetBatch(batch_size=batch_size, context=context_size, device=device)
# Define the object to estimate the loss
estim_loss = EstimateLoss(train_data=train, val_data=val)
# Selection of the model
# model = BiagramLanguageModel(vocab_size=vocab_size)
model = MultiHeadBiagramLanguageModel(
    vocab_size=vocab_size,
    context_size=context_size,
    emb_size=emb_size,
    head_size=head_size,
    head_dim=head_dim,
)
m = model.to(device)  # This generates a model whose call generates logits and losses
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
# Training iteration
for i in range(max_iters):
    # Print estimated loss
    if i % eval_interval == 0:
        avg_loss = estim_loss.calculate_loss(
            model=m, get_batch=get_batch, eval_iter=eval_iters
        )
        print(
            f'Iteration: {i}. Train loss = {avg_loss["train"]:.4f}, Val loss = {avg_loss["val"]:.4f}'
        )
    # Training
    xb, yb = get_batch(data=train)  # get batches
    logits, loss = m(xb, yb)  # compute logits and loss
    optimizer.zero_grad(set_to_none=True)  # set grads to none
    loss.backward()  # get grads
    optimizer.step()  # improve model parameters
# Get a prediction
predict = m.generate(
    input=torch.zeros((1, 1), dtype=torch.long, device=device),
    num_iterations=500,
)
print(file.decoder(predict[0].tolist()))
