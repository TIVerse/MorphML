"""API client library for MorphML.

Provides a Python client for interacting with the MorphML REST API.

Example:
    >>> from morphml.api.client import MorphMLClient
    >>> client = MorphMLClient("http://localhost:8000")
    >>> experiments = client.list_experiments()
    >>> exp = client.create_experiment("my-experiment", search_space={...})
"""

from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class MorphMLClient:
    """
    Python client for MorphML REST API.

    Provides convenient methods for all API endpoints.

    Attributes:
        base_url: Base URL of the API
        token: Optional authentication token
        session: Requests session

    Example:
        >>> client = MorphMLClient("http://localhost:8000")
        >>> client.login("user@example.com", "password")
        >>> experiments = client.list_experiments()
    """

    def __init__(self, base_url: str = "http://localhost:8000", token: Optional[str] = None):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the API
            token: Optional authentication token
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session = requests.Session()

        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

        logger.info(f"Initialized MorphML client for {base_url}")

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments

        Returns:
            Response JSON

        Raises:
            RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        except RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def login(self, username: str, password: str) -> str:
        """
        Login and get authentication token.

        Args:
            username: Username
            password: Password

        Returns:
            Access token

        Example:
            >>> token = client.login("user@example.com", "password")
        """
        response = self._request(
            "POST", "/api/v1/auth/login", json={"username": username, "password": password}
        )

        self.token = response["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

        logger.info(f"Logged in as {username}")

        return self.token

    # Experiment endpoints

    def create_experiment(
        self,
        name: str,
        search_space: Dict[str, Any],
        optimizer: str = "genetic",
        budget: int = 500,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            search_space: Search space configuration
            optimizer: Optimizer type
            budget: Evaluation budget
            config: Optional additional configuration

        Returns:
            Created experiment

        Example:
            >>> exp = client.create_experiment(
            ...     "cifar10-search",
            ...     search_space={"layers": [...]},
            ...     optimizer="genetic"
            ... )
        """
        return self._request(
            "POST",
            "/api/v1/experiments",
            json={
                "name": name,
                "search_space": search_space,
                "optimizer": optimizer,
                "budget": budget,
                "config": config or {},
            },
        )

    def list_experiments(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List experiments.

        Args:
            status: Optional status filter
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of experiments

        Example:
            >>> experiments = client.list_experiments(status="running")
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._request("GET", "/api/v1/experiments", params=params)

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment details

        Example:
            >>> exp = client.get_experiment("exp_abc123")
        """
        return self._request("GET", f"/api/v1/experiments/{experiment_id}")

    def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Start an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Updated experiment

        Example:
            >>> client.start_experiment("exp_abc123")
        """
        return self._request("POST", f"/api/v1/experiments/{experiment_id}/start")

    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Stop a running experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Updated experiment

        Example:
            >>> client.stop_experiment("exp_abc123")
        """
        return self._request("POST", f"/api/v1/experiments/{experiment_id}/stop")

    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Example:
            >>> client.delete_experiment("exp_abc123")
        """
        self._request("DELETE", f"/api/v1/experiments/{experiment_id}")
        logger.info(f"Deleted experiment: {experiment_id}")

    # Architecture endpoints

    def list_architectures(
        self,
        experiment_id: Optional[str] = None,
        min_fitness: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List architectures.

        Args:
            experiment_id: Optional experiment filter
            min_fitness: Optional minimum fitness filter
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of architectures

        Example:
            >>> archs = client.list_architectures(
            ...     experiment_id="exp_abc123",
            ...     min_fitness=0.9
            ... )
        """
        params = {"limit": limit, "offset": offset}
        if experiment_id:
            params["experiment_id"] = experiment_id
        if min_fitness is not None:
            params["min_fitness"] = min_fitness

        return self._request("GET", "/api/v1/architectures", params=params)

    def get_architecture(self, architecture_id: str) -> Dict[str, Any]:
        """
        Get architecture details.

        Args:
            architecture_id: Architecture ID

        Returns:
            Architecture details

        Example:
            >>> arch = client.get_architecture("arch_xyz789")
        """
        return self._request("GET", f"/api/v1/architectures/{architecture_id}")

    # Utility methods

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.

        Returns:
            Health status

        Example:
            >>> health = client.health_check()
            >>> print(health["status"])
        """
        return self._request("GET", "/health")

    def get_optimizers(self) -> List[Dict[str, Any]]:
        """
        Get list of available optimizers.

        Returns:
            List of optimizer information

        Example:
            >>> optimizers = client.get_optimizers()
            >>> for opt in optimizers:
            ...     print(opt["name"])
        """
        return self._request("GET", "/api/v1/optimizers")


def create_client(
    base_url: str = "http://localhost:8000",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> MorphMLClient:
    """
    Create and optionally authenticate API client.

    Args:
        base_url: Base URL of the API
        username: Optional username for authentication
        password: Optional password for authentication

    Returns:
        MorphMLClient instance

    Example:
        >>> client = create_client(
        ...     "http://localhost:8000",
        ...     "user@example.com",
        ...     "password"
        ... )
    """
    client = MorphMLClient(base_url)

    if username and password:
        client.login(username, password)

    return client
