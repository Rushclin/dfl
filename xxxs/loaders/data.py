import os
import gc
import logging
import concurrent.futures
from torch.utils.data import Subset, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale

from src import TqdmToLogger, stratified_split
from .split import split
from src.datasets import *

logger = logging.getLogger(__name__)


class SubsetWrapper(Dataset):
    """
    Wrapper class for PyTorch Subset that adds custom representation.
    
    This class allows us to create subsets of the dataset and provides 
    a custom suffix that can represent whether the dataset is for training or testing.
    """

    def __init__(self, subset, suffix):
        """
        Initialize the SubsetWrapper with a subset and a suffix.
        
        Args:
            subset (torch.utils.data.Subset): A subset of the original dataset.
            suffix (str): A string suffix to describe the subset (e.g., 'train', 'test').
        """
        self.subset = subset
        self.suffix = suffix

    def __getitem__(self, index):
        """
        Retrieve an item from the subset by its index.
        
        Args:
            index (int): Index of the item to retrieve.
            
        Returns:
            tuple: Input (image or data point) and its corresponding target (label).
        """
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        """
        Get the length of the subset (i.e., the number of items).
        
        Returns:
            int: The length of the subset.
        """
        return len(self.subset)

    def __repr__(self):
        """
        Custom string representation for the SubsetWrapper.
        
        Returns:
            str: Representation of the dataset with the custom suffix.
        """
        return f'{repr(self.subset.dataset.dataset)} {self.suffix}'


def load_dataset(args):
    """
    Load and split the dataset into training and testing sets.
    
    This function applies image transformations, splits the dataset for multiple clients, and 
    returns both the test dataset and the training dataset for clients.

    Args:
        args: The arguments containing settings for the dataset, such as resize dimensions, 
              split type, and number of clients.

    Returns:
        test_dataset (torch.utils.data.Dataset): The test dataset.
        client_datasets (list): A list containing subsets of the dataset for each client.
    """

    # Image transformation function to apply resizing, normalization, and grayscaling.
    def _get_transform(args):
        """
        Define the transformations applied to the dataset.
        
        Args:
            args: Arguments specifying transformations, like resizing and normalization.
            
        Returns:
            transform: A composed list of transformations for resizing, normalizing, 
                       and converting images to grayscale (specific for MNIST).
        """
        transform = Compose(
            [
                Resize((args.resize, args.resize)),  # Resize the image
                ToTensor(),  # Convert the image to a PyTorch tensor
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the tensor
                Grayscale(num_output_channels=1)  # Convert images to grayscale (for MNIST)
            ]
        )
        return transform

    # Function to construct dataset for each client, splitting it into training and testing sets.
    def _construct_dataset(train_dataset, idx, sample_indices):
        """
        Create subsets for each client and split the subset into training and testing sets.
        
        Args:
            train_dataset: The full training dataset.
            idx (int): Index representing the client.
            sample_indices: The indices for this client's subset.
        
        Returns:
            tuple: A tuple containing the training set and the testing set for the client.
        """
        subset = Subset(train_dataset, sample_indices)  # Create a subset from the dataset

        # Split the subset into training and testing sets using stratified sampling
        training_set, test_set = stratified_split(subset, args.test_size)

        # Wrap the training set and testing set with custom identifiers
        training_set = SubsetWrapper(
            training_set, f'< {str(idx).zfill(8)} > (train)')
        if len(subset) * args.test_size > 0:
            test_set = SubsetWrapper(
                test_set, f'< {str(idx).zfill(8)} > (test)')
        else:
            test_set = None
        return (training_set, test_set)

    # Initialize the training and test datasets
    train_dataset, test_dataset = None, None

    # Variables for split mapping and client datasets
    split_map, client_datasets = None, None

    # Prepare transformations for the datasets
    transforms = [None, None]
    transforms = [_get_transform(args), _get_transform(args)]  # Get transformations for training and testing

    # Fetch the actual datasets with the transformations applied
    train_dataset, test_dataset = fetch_dataset(args=args, transforms=transforms)

    # If we are working with local evaluation, check if we need to remove the test dataset
    if args.eval_type == 'local':
        if args.test_size == -1:
            assert test_dataset is not None  # Ensure test dataset exists
        test_dataset = None  # Set test_dataset to None if not needed

    # If no split map exists, split the dataset according to the specified strategy
    if split_map is None:
        logger.info(f'[SIMULATION] Distributing the dataset using the strategy: `{args.split_type.upper()}`!')
        split_map = split(args, train_dataset)  # Perform the dataset splitting
        logger.info(f'[SIMULATION] ...Finished distribution with the strategy: `{args.split_type.upper()}`!')

    # Create the dataset for clients if it hasn't been created yet
    if client_datasets is None:
        logger.info(f'[SIMULATION] Creating the dataset for clients!')

        client_datasets = []

        # Use a ThreadPoolExecutor to parallelize dataset creation for multiple clients
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() - 1)) as workhorse:
            for idx, sample_indices in TqdmToLogger(
                enumerate(split_map.values()),
                logger=logger,
                desc=f'[SIMULATION] ...Creating client datasets... ',
                total=len(split_map)
            ):
                # Submit dataset construction tasks for each client in parallel
                client_datasets.append(workhorse.submit(
                    _construct_dataset, train_dataset, idx, sample_indices).result())
        logger.info(f'[SIMULATION] ...Client dataset creation completed!')

    # Run the garbage collector to free up memory
    gc.collect() 

    return test_dataset, client_datasets
