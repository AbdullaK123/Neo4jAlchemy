# src/neo4jalchemy/orm/engine.py
import asyncio # Added import for asyncio.Lock
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, TRUST_ALL_CERTIFICATES
from typing import Optional, Tuple, Dict, Any, cast

class GraphEngine:
    """
    Represents the core interface to a Neo4j database, analogous to SQLAlchemy's Engine.

    It holds the configuration for connecting to the database and manages the
    underlying Neo4j AsyncDriver. An engine instance is typically created once per
    database configuration.
    """
    def __init__(
        self,
        uri: str,
        auth: Tuple[str, str],
        database: str = "neo4j", # Default database for sessions from this engine
        driver_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the GraphEngine. Does not establish a connection yet.
        Call `await engine.connect()` to establish the connection.

        Args:
            uri: The URI for the Neo4j instance (e.g., "bolt://localhost:7687").
            auth: A tuple of (username, password).
            database: The default Neo4j database name for sessions created by this engine.
            driver_config: Additional configuration options for the Neo4j driver.
        """
        self.uri: str = uri
        self.auth: Tuple[str, str] = auth
        self.default_database: str = database
        
        _driver_defaults = {
            "trusted_certificates": TRUST_ALL_CERTIFICATES, # For local dev; use proper certs in prod.
            "max_connection_lifetime": 3600 * 24 * 30,  # seconds
            "keep_alive": True,
            "user_agent": "Neo4jAlchemyEngine/0.1.0" # Can be overridden by driver_config
        }
        self.driver_config: Dict[str, Any] = {**_driver_defaults, **(driver_config or {})}
        
        self._driver: Optional[AsyncDriver] = None
        self._is_connected: bool = False
        self._connection_lock = asyncio.Lock() # To prevent race conditions on connect/close

    async def connect(self) -> None:
        """
        Establishes and verifies the connection to the Neo4j database using the
        engine's configuration. This method is idempotent.
        """
        async with self._connection_lock:
            if self._is_connected and self._driver:
                # print(f"GraphEngine: Already connected to {self.uri}.")
                return

            print(f"GraphEngine: Connecting to {self.uri} (default session DB: '{self.default_database}')...")
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=self.auth,
                    **self.driver_config
                )
                await self._driver.verify_connectivity()
                self._is_connected = True
                print(f"GraphEngine: Successfully connected to {self.uri}.")
            except Exception as e:
                self._driver = None # Ensure driver is None on failure
                self._is_connected = False
                print(f"GraphEngine: Connection to {self.uri} failed: {e}")
                raise ConnectionError(f"Failed to connect to Neo4j at {self.uri}: {e}") from e

    async def close(self) -> None:
        """Closes the Neo4j driver connection if it's open."""
        async with self._connection_lock:
            if self._driver and self._is_connected:
                print(f"GraphEngine: Closing connection to {self.uri}...")
                await self._driver.close()
                self._driver = None
                self._is_connected = False
                print(f"GraphEngine: Connection to {self.uri} closed.")
            elif self._driver and not self._is_connected:
                # This state might occur if connect failed mid-way but driver was assigned
                print(f"GraphEngine: Driver for {self.uri} exists but was not fully connected. Attempting close.")
                await self._driver.close() # Attempt to close even if verify_connectivity failed
                self._driver = None
                self._is_connected = False # Ensure this is false
            # else:
                # print(f"GraphEngine: No active or initialized driver for {self.uri} to close.")


    def get_session(self, database: Optional[str] = None) -> AsyncSession:
        """
        Returns an asynchronous Neo4j session from the engine's driver.

        Args:
            database: The name of the database to use for this session.
                      If None, uses the engine's `default_database`.

        Returns:
            An AsyncSession object.

        Raises:
            ConnectionError: If the engine is not connected. Call `await engine.connect()` first.
        """
        if not self._driver or not self._is_connected:
            raise ConnectionError(
                f"GraphEngine for {self.uri} is not connected. Call `await engine.connect()` first."
            )
        
        db_to_use = database or self.default_database
        return cast(AsyncSession, self._driver.session(database=db_to_use))

    @property
    def driver(self) -> AsyncDriver:
        """
        Provides direct access to the underlying Neo4j AsyncDriver.
        Use with caution. Prefer using `engine.get_session()`.

        Raises:
            ConnectionError: If the engine is not connected.
        """
        if not self._driver or not self._is_connected:
            raise ConnectionError(
                f"GraphEngine for {self.uri} is not connected. Call `await engine.connect()` first."
            )
        return self._driver

    @property
    def connected(self) -> bool:
        """Returns True if the engine is currently connected, False otherwise."""
        return self._is_connected

    async def __aenter__(self) -> "GraphEngine":
        """Allows the engine to be used as an async context manager for connect/close."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Ensures the engine's connection is closed when exiting the context."""
        await self.close()


def create_graph_engine(
    uri: str,
    auth: Tuple[str, str],
    database: str = "neo4j", # Default database for sessions created from this engine
    **driver_config: Any # Pass driver_config as keyword arguments
) -> GraphEngine:
    """
    Creates and returns a GraphEngine instance.
    This is the primary setup function, analogous to SQLAlchemy's create_engine.
    The engine instance encapsulates the database connection configuration and
    manages the underlying Neo4j driver.

    The engine must be explicitly connected using `await engine.connect()`
    or by using it as an async context manager (`async with engine:`).

    Args:
        uri: The URI for the Neo4j instance (e.g., "bolt://localhost:7687").
        auth: A tuple of (username, password).
        database: The default Neo4j database name for sessions that will eventually
                  be created using this engine.
        **driver_config: Additional configuration options for the Neo4j driver
                         (e.g., user_agent, keep_alive, max_connection_pool_size).

    Returns:
        A GraphEngine instance.
    """
    print(f"Neo4jAlchemy: Creating GraphEngine for URI: {uri}, Default DB for its sessions: {database}")
    return GraphEngine(uri=uri, auth=auth, database=database, driver_config=driver_config)
