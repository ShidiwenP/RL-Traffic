import numpy as np

# Load the data
training_queues = np.load("dqn_queue_history.npy")
eval_queues = np.load("dqn_eval_queue_history.npy")

# Print summary statistics
print(f"Training - Mean: {training_queues.mean():.2f}, Min: {training_queues.min():.2f}, Max: {training_queues.max():.2f}")
print(f"Evaluation - Mean: {eval_queues.mean():.2f}, Min: {eval_queues.min():.2f}, Max: {eval_queues.max():.2f}")