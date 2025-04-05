from src.utils import eval_model

experiment = {
    "training_set": "progan",
    "batch_size": 32,
    "savpath": "results/flow",
}

eval_model(
    experiment=experiment,
    ncls=1,
    num_steps=8,
)