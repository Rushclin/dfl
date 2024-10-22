import os
import gc
import json
import torch
import random
import logging
import concurrent.futures

from collections import ChainMap, defaultdict

from src import init_weights, TqdmToLogger, MetricManager, Client
from .basenode import BaseNode

logger = logging.getLogger(__name__)

class Node(BaseNode):
    """
    Node class represents either a client or a server node in a decentralized federated learning system.
    Handles model initialization, client creation, sampling, aggregation, and training rounds.
    """
    def __init__(self, args, writer, client_datasets, server_dataset, model):
        """
        Initialize the node (either client or server).
        
        Args:
            args: Arguments specifying configuration (e.g., learning rate, number of clients).
            writer: Writer for logging metrics (e.g., TensorBoard).
            client_datasets: The datasets available for each client.
            server_dataset: The dataset available on the server (for evaluation).
            model: The initial global model for the federated learning.
        """
        super(Node, self).__init__()
        self.args = args
        self.writer = writer
        self.server_dataset = server_dataset
        self.round = 0
        self.global_model = self._init_model(model)  # Initialize the global model
        self.opt_kwargs = dict(lr=self.args.lr, momentum=self.args.beta1)
        self.curr_lr = self.args.lr
        self.clients = self._create_clients(client_datasets)  # Create client nodes
        self.results = defaultdict(dict)

    def _init_model(self, model):
        """
        Initialize the global model by setting weights and logging the process.
        
        Args:
            model: The PyTorch model to initialize.
            
        Returns:
            model: Initialized model with set weights.
        """
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Initializing model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...Model initialization complete ({self.args.model_name})!')
        return model

    def _create_clients(self, client_datasets):
        """
        Create client nodes from datasets provided for each client.
        
        Args:
            client_datasets: A list of datasets for each client.
            
        Returns:
            clients: A list of created client nodes.
        """
        def __create_client(identifier, datasets):
            client = Client(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logging.info(f'[Round: {str(self.round).zfill(4)}] Creating clients!')
        clients = []
        # Parallel client creation using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets),
                logger=logger,
                desc=f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...Creating clients... ',
                total=len(client_datasets)
            ):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...{self.args.K} clients created!')
        return clients

    def _sample_clients(self, exclude=[]):
        """
        Sample a subset of clients for the current training round.
        
        Args:
            exclude: List of clients to exclude from sampling.
            
        Returns:
            sampled_client_ids: List of sampled client IDs.
        """
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Sampling clients!')
        if exclude == []:
            # Sample clients based on a fraction 'C' of total clients 'K'
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))
        else:
            # Sample from the remaining clients who were not excluded
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0:
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                sampled_client_ids = sorted(random.sample([identifier for identifier in [
                    i for i in range(self.args.K)] if identifier not in exclude], num_sampled_clients))
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients sampled!')
        return sampled_client_ids

    def _elect_server(self, sampled_client_ids):
        """
        Elect a client to act as the server for the current iteration.
        
        Args:
            sampled_client_ids: List of sampled client IDs.
            
        Returns:
            elected_client_id: The ID of the elected server client.
        """
        elected_client_id = random.choice(sampled_client_ids)  # Randomly elect a client
        logger.info(f"Client {elected_client_id} has been elected as the server for this round.")
        return elected_client_id

    def _request(self, ids, eval=False):
        """
        Send requests to the clients for model updates or evaluations.
        
        Args:
            ids: List of client IDs to request updates/evaluations from.
            eval: Boolean flag to indicate if it's an evaluation or update request.
            
        Returns:
            updated_sizes: A dictionary with the number of training examples for each client.
            results: A dictionary with the update or evaluation results from each client.
        """
        def __update_clients(client):
            if client.model is None:
                client.download(self.global_model)
            client.args.lr = self.curr_lr
            update_result = client.update(self.round)
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client):
            if client.model is None:
                client.download(self.global_model)
            eval_result = client.evaluate()
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Requesting {"evaluation" if eval else "updates"} from clients...')
        jobs, results = [], []
        # Parallelize requests using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
            for idx in TqdmToLogger(ids, logger=logger,
                                    desc=f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Requesting client {"evaluation" if eval else "updates"}... ',
                                    total=len(ids)):
                if eval:
                    jobs.append(workhorse.submit(__evaluate_clients, self.clients[idx]))
                else:
                    jobs.append(workhorse.submit(__update_clients, self.clients[idx]))
            for job in concurrent.futures.as_completed(jobs):
                results.append(job.result())
        return dict(ChainMap(*list(map(list, zip(*results)))[0])), dict(ChainMap(*list(map(list, zip(*results)))[1]))

    
    def _aggregate(self, elected_client, ids, updated_sizes):
        """
        Aggregate the models from all clients on the elected client (acting as server).
        
        Args:
            elected_client: The client acting as the server for this round.
            ids: List of client IDs that participated in the update.
            updated_sizes: Dictionary of the number of training samples for each client.
            
        Returns:
            aggregated_model: The aggregated global model.
        """
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregating models on client {elected_client.id}...')
        
        total_data_points = sum(updated_sizes.values())
        
        # Initialize the aggregated model with zeros for each parameter dynamically
        aggregated_model = {key: torch.zeros_like(param) for key, param in self.global_model.state_dict().items()}

        # The elected client aggregates models from others
        for client_id in ids:
            client = self.clients[client_id]
            weight = updated_sizes[client_id] / total_data_points  # Weight each client's contribution
            local_model = client.model.state_dict()

            # Aggregate each parameter in the state_dict
            for key, param in local_model.items():
                aggregated_model[key] += weight * param

        # Load the aggregated model into the elected client's model
        elected_client.model.load_state_dict(aggregated_model)
        
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregation complete on client {elected_client.id}.')
        
        return aggregated_model


    def _send_global_model(self, elected_client):
        """
        Send the aggregated global model from the elected client to all other clients.
        
        Args:
            elected_client: The client that acted as the server for the round.
        """
        for client in self.clients:
            if client.id != elected_client.id:
                client.download(elected_client.model)
        logger.info(f"Client {elected_client.id} has sent the aggregated global model to all clients.")

    def train_round(self):
        """
        Perform one round of training, including client sampling, server election, 
        model aggregation, and global model distribution.
        
        Returns:
            sampled_client_ids: The IDs of the clients that participated in this round.
        """
        # Step 1: Sample clients for the round
        sampled_client_ids = self._sample_clients()

        # Step 2: Elect a client to act as the server
        elected_client_id = self._elect_server(sampled_client_ids)
        elected_client = self.clients[elected_client_id]

        # Step 3: Request updates from clients
        updated_sizes, _ = self._request(sampled_client_ids, eval=False)

        # Step 4: Aggregate the models on the elected client
        aggregated_model = self._aggregate(elected_client, sampled_client_ids, updated_sizes)

        # Step 5: Send the aggregated model to all clients
        self._send_global_model(elected_client)

        # Step 6: Update the global model and move to the next round
        self.global_model.load_state_dict(aggregated_model)
        self.round += 1
        logger.info(f"Round {self.round} completed.")
        return sampled_client_ids

    def evaluate(self, excluded_ids):
        """
        Evaluate the global model on the clients, excluding certain clients if necessary.
        
        Args:
            excluded_ids: List of client IDs to exclude from evaluation.
        """
        selected_ids = self._sample_clients(exclude=excluded_ids)
        _ = self._request(selected_ids, eval=True)
        self._central_evaluate()

    @torch.no_grad()
    def _central_evaluate(self):
        """
        Centrally evaluate the global model on the server dataset and log results.
        """
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.eval()
        self.global_model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.global_model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)
            mm.track(loss.item(), outputs, targets)

        self.global_model.to('cpu')
        mm.aggregate(len(self.server_dataset))
        result = mm.results

        server_log_string = f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Server Evaluation: '
        loss = result['loss']
        server_log_string += f'| Loss: {loss:.4f} '
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        self.writer.add_scalar('Server Loss', loss, self.round)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        self.writer.flush()
        self.results[self.round]['server_evaluated'] = result

    def finalize(self):
        """
        Finalize the training process by saving the global model and results.
        """
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Saving final model!')
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file:
            json.dump(self.results, result_file, indent=4)
        torch.save(self.global_model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt'))
        self.writer.close()
        logger.info(f'[{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...Training completed!')
