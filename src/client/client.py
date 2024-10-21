import sys
import copy
import torch
import logging
import inspect
import itertools

from .baseclient import BaseClient
from src import MetricManager

logger = logging.getLogger(__name__)


class Client(BaseClient):
    def __init__(self, args, training_set, test_set):
        """
        Initializes a client with its training and testing datasets, as well as optimizer and loss criterion.
        Args:
            args: Configuration and arguments for the client.
            training_set: The dataset for training the local model.
            test_set: The dataset for evaluating the local model.
        """
        super(Client, self).__init__()
        self.args = args

        self.training_set = training_set
        self.test_set = test_set

        print(f"TEST ====> {test_set}")

        # Set up the optimizer and criterion (loss function) based on provided arguments.
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        # Create data loaders for training and testing datasets.
        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

    def _refine_optim_args(self, args):
        """
        Refines the optimizer arguments based on the available attributes in args.
        Only includes the arguments required by the optimizer.
        Args:
            args: Configuration and arguments passed to the optimizer.
        Returns:
            refined_args: A dictionary containing only the relevant arguments for the optimizer.
        """
        required_args = inspect.getfullargspec(self.optim)[0]  # Get required arguments for the optimizer

        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        """
        Creates a PyTorch DataLoader for the client's dataset.
        Args:
            dataset: The dataset to create the DataLoader from (training or test set).
            shuffle: Whether to shuffle the dataset before creating the DataLoader.
        Returns:
            A PyTorch DataLoader object.
        """
        if self.args.B == 0:  # If batch size is 0, set it to the size of the dataset
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def update(self, round: int):
        """
        Performs local model training for the client over multiple epochs.
        Args:
            round: The current round number in federated learning.
        Returns:
            mm.results: The training metrics tracked and aggregated by MetricManager.
        """
        mm = MetricManager(self.args.eval_metrics)  # Track metrics such as loss and accuracy
        self.model.train()
        self.model.to(self.args.device)

        # Initialize optimizer with refined arguments
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

        for e in range(self.args.E):  # Loop through epochs
            logger.info(f'[EPOCH] Round {e} / {self.args.E}')
            for inputs, targets in self.train_loader:  # Iterate through training data
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion()(outputs, targets)  # Compute loss

                for param in self.model.parameters():
                    param.grad = None  # Zero gradients manually for each parameter
                loss.backward()  # Backward pass
                if self.args.max_grad_norm > 0:
                    # Clip gradients to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()  # Update the model's parameters

                mm.track(loss.item(), outputs, targets)  # Track loss and output metrics

            mm.aggregate(len(self.training_set), e + 1)  # Aggregate metrics for the epoch

        self.model.to('cpu')  # Move model back to CPU after training
        return mm.results

    @torch.inference_mode()
    def evaluate(self):
        """
        Evaluates the client's local model on the test dataset.
        This method runs in inference mode (no gradient calculation).
        Returns:
            A dictionary containing the evaluation results (loss and other metrics).
        """
        if self.args.train_only:  # If only training is required, return dummy results
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.args.device)

        # Evaluate the model on the test set
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)  # Compute loss

            mm.track(loss.item(), outputs, targets)  # Track evaluation metrics
        else:
            self.model.to('cpu')  # Move model back to CPU after evaluation
            mm.aggregate(len(self.test_set))  # Aggregate evaluation metrics
        return mm.results

    def download(self, model):
        """
        Downloads the global model to the client.
        Args:
            model: The global model to be copied to the client.
        """
        self.model = copy.deepcopy(model)

    def upload(self):
        """
        Uploads the local model's parameters and buffers.
        Returns:
            An iterator containing the model's named parameters and buffers.
        """
        return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])

    def get_model_size(model):
        """
        Calculates and returns the size of the model in megabytes (MB).
        Args:
            model: The model to compute the size of.
        Returns:
            model_size_mb: The size of the model in megabytes.
        """
        model_size_bytes = sys.getsizeof(model)
        model_size_mb = model_size_bytes / (1024 * 1024)  # Convert bytes to MB
        return model_size_mb

    def __len__(self):
        """
        Returns the length of the client's training set.
        Returns:
            The number of data samples in the training set.
        """
        return len(self.training_set)

    def __repr__(self):
        """
        Returns a string representation of the client.
        Returns:
            A formatted string indicating the client's ID.
        """
        return f'CLIENT < {self.id} >'
